import { db } from '../db';

const MS_PER_DAY = 24 * 60 * 60 * 1000;

// ─── Data Gathering Helpers ────────────────────────────────────────────────

async function computeSalesVelocity(productId) {
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * MS_PER_DAY);
    const thirtyDaysAgo = new Date(now.getTime() - 30 * MS_PER_DAY);

    const sales = await db.sales.toArray();
    let units7 = 0;
    let units30 = 0;
    let revenue7 = 0;

    for (const sale of sales) {
        if (!sale.date) continue;
        const date = new Date(sale.date);
        const items = Array.isArray(sale.items) ? sale.items : [];
        const productItems = items.filter(i => i.id === productId);

        for (const item of productItems) {
            const qty = parseFloat(item.quantity) || 0;
            const price = parseFloat(item.price) || 0;
            if (date >= thirtyDaysAgo) units30 += qty;
            if (date >= sevenDaysAgo) { units7 += qty; revenue7 += qty * price; }
        }
    }

    return { units7, units30, revenue7 };
}

async function getEarliestExpiry(productId, productFallbackExpiry) {
    const batches = await db.batches.where('product_id').equals(productId).toArray();
    const dates = [];
    for (const b of batches) {
        if (b.expiry_date) {
            const d = new Date(b.expiry_date);
            if (!isNaN(d.getTime())) dates.push(d);
        }
    }
    if (dates.length > 0) return new Date(Math.min(...dates.map(d => d.getTime())));
    if (productFallbackExpiry) {
        const d = new Date(productFallbackExpiry);
        if (!isNaN(d.getTime())) return d;
    }
    return null;
}

// ─── Fallback: Rule-Based Engine (used if Gemini is unavailable) ──────────

function calculateFallbackPricing(product, velocity, expiryDate) {
    const now = new Date();
    const currentPrice = parseFloat(product.price || 0);
    const costPrice = parseFloat(product.cost || 0);
    let discountPct = 0;
    let reason = 'Stable demand — no adjustment needed.';

    if (expiryDate) {
        const daysToExpiry = (expiryDate.getTime() - now.getTime()) / MS_PER_DAY;
        if (daysToExpiry < 0) { discountPct = 90; reason = 'EXPIRED — immediate clearance required.'; }
        else if (daysToExpiry <= 7) { discountPct = 40; reason = 'Expiring in less than a week.'; }
        else if (daysToExpiry <= 30) { discountPct = 20; reason = 'Expiring within 30 days.'; }
        else if (daysToExpiry <= 60) { discountPct = 10; reason = 'Expiring within 2 months.'; }
    }

    if (discountPct === 0) {
        const stock = parseInt(product.stock_quantity || 0, 10);
        if (stock > 20 && velocity.units30 < 5) { discountPct = 15; reason = 'High stock with very low monthly sales.'; }
        else if (stock > 10 && velocity.units7 === 0) { discountPct = 5; reason = 'No sales this week — slow mover.'; }
    }

    let suggestedPrice = currentPrice * (1 - discountPct / 100);
    const costFloor = costPrice * 1.05;
    if (suggestedPrice < costFloor) {
        suggestedPrice = costFloor;
        discountPct = currentPrice > 0 ? Math.max(0, ((currentPrice - suggestedPrice) / currentPrice) * 100) : 0;
        reason = 'Price protected to maintain minimum 5% profit margin.';
    }

    return {
        suggested_price: parseFloat(suggestedPrice.toFixed(2)),
        suggested_discount_pct: Math.round(discountPct),
        reason: String(reason),
        calculation_method: 'Rule-Based Fallback Engine'
    };
}

// ─── Gemini AI Pricing Engine ─────────────────────────────────────────────

async function getAIPricingSuggestion(product, velocity, expiryDate) {
    const rawStoredKey = localStorage.getItem('GEMINI_API_KEY');
    const apiKey = (rawStoredKey && rawStoredKey.trim() !== '') 
        ? rawStoredKey.trim() 
        : 'AIzaSyB3D74D6wSgG2rt4syScqWMkYYUe-vJcIc';
    
    const now = new Date();

    const daysToExpiry = expiryDate
        ? Math.round((expiryDate.getTime() - now.getTime()) / MS_PER_DAY)
        : null;

    const marginPct = (parseFloat(product.cost) > 0)
        ? (((parseFloat(product.price) - parseFloat(product.cost)) / parseFloat(product.cost)) * 100).toFixed(1)
        : '0';

    const prompt = `As a retail pricing expert, suggest a price for: ${String(product.name)}.
Current Price: ${product.price} LKR
Cost: ${product.cost} LKR
Margin: ${marginPct}%
Stock: ${product.stock_quantity || 0} unit(s)
Recent Sales (7/30 days): ${velocity.units7}/${velocity.units30} units.
${daysToExpiry !== null ? `Expires in: ${daysToExpiry} days.` : 'No expiry date.'}

Rule: Min 5% profit margin must be maintained.
Respond with ONLY this format:
PRICE: <number>
DISCOUNT: <number>
REASON: <short text>`;

    try {
        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=${apiKey}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [{ role: 'user', parts: [{ text: prompt }] }],
                    generationConfig: {
                        temperature: 0.1,
                        maxOutputTokens: 256,
                        topP: 0.8
                    }
                })
            }
        );

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err?.error?.message || `API error ${response.status}`);
        }

        const data = await response.json();
        
        let rawBody = '';
        if (data && data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts && data.candidates[0].content.parts[0]) {
            rawBody = String(data.candidates[0].content.parts[0].text || '');
        }

        if (rawBody.length < 5) {
            throw new Error('AI returned an empty or insufficient response.');
        }

        // Extremely safe parsing
        const cleanText = rawBody.replace(/[\*#]/g, '').trim();

        const findVal = (label) => {
            const regex = new RegExp(`${String(label)}:?\\s*(?:Rs\\.?|LKR)?\\s*([\\d,.]+)`, 'i');
            const match = cleanText.match(regex);
            if (match && match[1]) {
                return String(match[1]).replace(/,/g, '');
            }
            return null;
        };

        let priceStr = findVal('PRICE|SUGGESTED_PRICE|SUGGESTED');
        let discountStr = findVal('DISCOUNT|DISCOUNT_PERCENT');

        // Fallback search for any numbers
        if (!priceStr) {
            const anyNums = cleanText.match(/[\d,.]+/g);
            if (anyNums && anyNums.length > 0) {
                priceStr = String(anyNums[0] || '').replace(/,/g, '');
                if (!discountStr && anyNums.length > 1) {
                    discountStr = String(anyNums[1] || '').replace(/,/g, '');
                }
            }
        }

        if (!priceStr) {
            throw new Error(`Invalid format in AI response. Received: ${rawBody.slice(0, 40)}`);
        }

        const suggestedPriceValue = parseFloat(priceStr);
        const suggestedDiscountValue = parseInt(discountStr || '0');

        // Reason extraction
        const reasonMatch = cleanText.match(/(?:REASON|WHY|BECAUSE):?\\s*(.+)/i);
        const reasonStr = (reasonMatch && reasonMatch[1]) 
            ? String(reasonMatch[1]).split('\n')[0].trim().slice(0, 80)
            : 'Optimization based on inventory and sales data';

        // Final Safety Check
        const costFloor = (parseFloat(product.cost) || 0) * 1.05;
        const finalPrice = Math.max(suggestedPriceValue, costFloor);

        return {
            suggested_price: parseFloat(finalPrice.toFixed(2)),
            suggested_discount_pct: Math.min(90, Math.max(0, suggestedDiscountValue)),
            reason: finalPrice > suggestedPriceValue ? `${reasonStr} (Price protected)` : reasonStr,
            calculation_method: 'Gemini AI Pricing Engine'
        };
    } catch (e) {
        console.error('Gemini Pricing Logic Error:', e);
        throw e;
    }
}

// ─── Public Service ────────────────────────────────────────────────────────

export const pricingService = {
    isAvailable() {
        return true;
    },

    async getSuggestionForProduct(productId) {
        const product = await db.products.get(productId);
        if (!product) throw new Error('Product not found in database');

        const [velocity, expiryDate] = await Promise.all([
            computeSalesVelocity(productId),
            getEarliestExpiry(productId, product.expiry_date),
        ]);

        try {
            return await getAIPricingSuggestion(product, velocity, expiryDate);
        } catch (err) {
            // Surface specific error captured in catch-all for troubleshooting
            const errorMsg = String(err && err.message ? err.message : err || 'Unknown AI error');
            console.warn('AI Suggester failed, using fallback:', errorMsg);

            const fallback = calculateFallbackPricing(product, velocity, expiryDate);
            fallback.calculation_method = `Rule-Based Fallback (${errorMsg})`;
            return fallback;
        }
    },
};
