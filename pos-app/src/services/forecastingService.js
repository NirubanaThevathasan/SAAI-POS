import * as tf from '@tensorflow/tfjs';
import { db } from '../db';
import { aiService } from './aiService';

/**
 * ForecastingService handles AI-driven time-series predictions and demand analysis.
 * Now enhanced with Gemini for 'pre-trained' intelligence fallback.
 */
class ForecastingService {
    /**
     * Prepares historical daily revenue data for time-series modeling.
     */
    async getDailyRevenueData(days = 90) {
        const sales = await db.sales.toArray();
        const dailyTotals = {};

        sales.forEach(sale => {
            let saleDate;
            if (sale.date instanceof Date) {
                saleDate = sale.date;
            } else if (!isNaN(Number(sale.date))) {
                saleDate = new Date(Number(sale.date));
            } else {
                saleDate = new Date(sale.date);
            }
            
            const date = saleDate.toISOString().split('T')[0];

            const amount = Number(sale.total || sale.grand_total || sale.grandTotal || 0);
            dailyTotals[date] = (dailyTotals[date] || 0) + amount;
        });

        // Fill gaps with 0
        const sortedDates = Object.keys(dailyTotals).sort();
        if (sortedDates.length === 0) return [];

        const start = new Date(sortedDates[0]);
        const end = new Date();
        const data = [];

        for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
            const dateStr = d.toISOString().split('T')[0];
            data.push({
                date: dateStr,
                value: dailyTotals[dateStr] || 0
            });
        }

        return data.slice(-days);
    }

    /**
     * Predicts next day revenue using an LSTM model with Gemini fallback.
     */
    async predictNextDayRevenue(mode = 'revenue', historyDays = 30) {
        try {
            const rawData = await this.getDailyRevenueData(historyDays + 7);
            if (rawData.length < 5) return 0;

            // If predicting profit, we need to adjust the values
            let values;
            if (mode === 'profit') {
                const sales = await db.sales.toArray();
                const dailyProfit = {};
                sales.forEach(s => {
                    let saleDate;
                    if (s.date instanceof Date) saleDate = s.date;
                    else if (!isNaN(Number(s.date))) saleDate = new Date(Number(s.date));
                    else saleDate = new Date(s.date);
                    const dateStr = saleDate.toISOString().split('T')[0];
                    
                    const rev = Number(s.grandTotal || s.total || s.grand_total || 0);
                    let cost = 0;
                    if (s.items && s.items.length > 0) {
                        s.items.forEach(i => cost += (Number(i.cost || 0) * Number(i.quantity || 0)));
                    }
                    
                    // FALLBACK: If cost is 0 (missing data), assume a 30% profit margin as a baseline
                    if (cost <= 0 && rev > 0) {
                        cost = rev * 0.7; // Estimates profit at 30%
                    }
                    
                    dailyProfit[dateStr] = (dailyProfit[dateStr] || 0) + (rev - cost);
                });
                values = rawData.map(d => dailyProfit[d.date] || 0);
            } else {
                values = rawData.map(d => d.value);
            }

            if (rawData.length < 14) return await this.getGeminiForecast(rawData, mode);

            // Simple normalization (Min-Max)
            const max = Math.max(...values, 1);
            const min = Math.min(...values);
            const normalized = values.map(v => (v - min) / (max - min || 1));

            const windowSize = Math.min(7, Math.floor(normalized.length / 2));
            const X = [], Y = [];
            for (let i = 0; i < normalized.length - windowSize; i++) {
                X.push(normalized.slice(i, i + windowSize));
                Y.push(normalized[i + windowSize]);
            }

            if (X.length === 0) return await this.getGeminiForecast(rawData, mode);

            const inputTensor = tf.tensor2d(X, [X.length, windowSize]).reshape([X.length, windowSize, 1]);
            const labelTensor = tf.tensor2d(Y, [Y.length, 1]);

            const model = tf.sequential();
            model.add(tf.layers.lstm({ units: 32, inputShape: [windowSize, 1], returnSequences: false }));
            model.add(tf.layers.dense({ units: 1 }));
            model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

            await model.fit(inputTensor, labelTensor, { epochs: 50, verbose: 0 });

            const latestWindow = normalized.slice(-windowSize);
            const input = tf.tensor3d([latestWindow.map(v => [v])], [1, windowSize, 1]);
            const prediction = model.predict(input);
            const predictedValue = (await prediction.data())[0];

            tf.dispose([inputTensor, labelTensor, input, prediction]);

            const result = predictedValue * (max - min) + min;
            const recentAvg = values.slice(-3).reduce((a, b) => a + b, 0) / 3;
            if (result <= 0 && recentAvg > 0) return await this.getGeminiForecast(rawData, mode);

            return isNaN(result) ? await this.getGeminiForecast(rawData, mode) : Math.max(0, result);
        } catch (err) {
            console.error("Forecasting Error:", err);
            return 0;
        }
    }

    /**
     * Uses Gemini API as a pre-trained time-series model.
     */
    async getGeminiForecast(history, mode = 'revenue') {
        if (!aiService.hasApiKey()) return history[history.length - 1]?.value || 0;

        const dataPoints = history.map(h => `${h.date}: ${h.value}`).join('\n');
        const prompt = `Analyze this daily ${mode} history and predict the ${mode} for the NEXT DAY ONLY. 
        RESPOND ONLY WITH THE NUMBER.
        DATA:
        ${dataPoints}`;

        try {
            const response = await aiService.ask(prompt);
            const match = response.match(/[\d.]+/);
            const predicted = match ? parseFloat(match[0]) : 0;
            
            // If Gemini fails or returns 0, use the last known sale value as a baseline
            if (predicted <= 0) {
                return history.find(h => h.value > 0)?.value || 0;
            }
            return predicted;
        } catch (e) {
            // Final safety: Use the most recent non-zero sale
            const reversed = [...history].reverse();
            const lastActive = reversed.find(h => h.value > 0);
            return lastActive ? lastActive.value : 0;
        }
    }

    /**
     * Simplified ARIMA-like forecasting (Exponential Smoothing).
     */
    async predictWeeklyRevenue() {
        const data = await this.getDailyRevenueData(60);
        
        // If NO data at all, show empty stats rather than crash
        if (data.length === 0) return Array(7).fill(0).map((_, i) => ({ day: i + 1, value: 0 }));

        if (data.length < 21) {
            // "Damped Growth" Fallback for new businesses
            const values = data.map(d => d.value);
            const recentAvg = values.slice(-3).reduce((a, b) => a + b, 0) / 3;
            // Cap growth at 5% per day max if data is insecure
            return Array(7).fill(0).map((_, i) => ({ 
                day: i + 1, 
                value: Math.max(0, recentAvg * (1 + (i * 0.02))) 
            }));
        }

        const values = data.map(d => d.value);
        const max = Math.max(...values, 1), min = Math.min(...values);
        const normalized = values.map(v => (v - min) / (max - min || 1));
        const windowSize = 7;
        
        // Build model once for the week
        const X = [], Y = [];
        for (let i = 0; i < normalized.length - windowSize; i++) {
            X.push(normalized.slice(i, i + windowSize));
            Y.push(normalized[i + windowSize]);
        }
        
        const model = tf.sequential();
        model.add(tf.layers.lstm({ units: 16, inputShape: [windowSize, 1] }));
        model.add(tf.layers.dense({ units: 1 }));
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        await model.fit(tf.tensor3d(X.map(x => x.map(v => [v])), [X.length, windowSize, 1]), tf.tensor2d(Y, [Y.length, 1]), { epochs: 20, verbose: 0 });

        const forecasts = [];
        let currentWindow = [...normalized.slice(-windowSize)];

        for (let i = 0; i < 7; i++) {
            const input = tf.tensor3d([currentWindow.map(v => [v])], [1, windowSize, 1]);
            const pred = model.predict(input);
            const val = (await pred.data())[0];
            const denormalized = Math.max(0, val * (max - min) + min);
            
            forecasts.push({ day: i + 1, value: denormalized });
            currentWindow.push(val);
            currentWindow.shift();
            tf.dispose([input, pred]);
        }
        
        tf.dispose(model);
        return forecasts;
    }

    /**
     * Demand Analysis: Predict Slow vs Fast moving items.
     */
    async getDemandInsights() {
        const sales = await db.sales.toArray();
        const products = await db.products.toArray();
        const stockArr = await db.stock.toArray();

        // 30 day window for velocity calculation
        const now = new Date();
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(now.getDate() - 30);

        const productSales = {};

        // Aggregate totals with Universal Date Support
        sales.forEach(sale => {
            let saleDate;
            if (sale.date instanceof Date) saleDate = sale.date;
            else if (!isNaN(Number(sale.date))) saleDate = new Date(Number(sale.date));
            else saleDate = new Date(sale.date);
            
            if (saleDate >= thirtyDaysAgo && sale.items) {
                sale.items.forEach(item => {
                    const pid = item.product_id || item.id;
                    if (pid) {
                        productSales[pid] = (productSales[pid] || 0) + (Number(item.quantity) || 0);
                    }
                });
            }
        });

        // Calculate Average Sales across all active products for dynamic thresholds
        const activeSalesValues = Object.values(productSales);
        const avgSales = activeSalesValues.length > 0 
            ? activeSalesValues.reduce((a, b) => a + b, 0) / activeSalesValues.length 
            : 0;

        const insights = products.map(p => {
            const totalSold = productSales[p.id] || 0;
            const liveStock = stockArr
                .filter(s => s.product_id === p.id)
                .reduce((sum, s) => sum + (Number(s.quantity) || 0), 0);

            // Dynamic Categorization based on relative performance
            let status = 'Normal Moving';
            let color = '#6366f1'; // Indigo

            if (totalSold > (avgSales * 1.5) && totalSold >= 3) {
                status = '🔥 Fast Moving';
                color = '#10b981'; // Emerald
            } else if (totalSold === 0 && liveStock > 0) {
                status = '🧊 Dead Stock';
                color = '#ef4444'; // Rose
            } else if (totalSold > 0 && totalSold < (avgSales * 0.5)) {
                status = '🐢 Slow Moving';
                color = '#f59e0b'; // Amber
            } else if (totalSold > 0 && liveStock < (totalSold * 0.5)) {
                status = '⚠️ Stockout Risk';
                color = '#f97316'; // Orange
            }

            return {
                id: p.id,
                name: p.name,
                totalSold,
                currentStock: liveStock,
                status,
                color
            };
        });

        return insights.sort((a, b) => b.totalSold - a.totalSold);
    }

    /**
     * Cash Flow Forecasting (Revenue vs Expenses).
     */
    async getCashFlowForecast() {
        const dailyRev = await this.getDailyRevenueData(30);
        const expenses = await db.expenses.toArray();

        const dailyExp = {};
        expenses.forEach(e => {
            const date = e.date instanceof Date
                ? e.date.toISOString().split('T')[0]
                : String(e.date).split('T')[0];
            dailyExp[date] = (dailyExp[date] || 0) + Number(e.amount || 0);
        });

        return dailyRev.map(d => ({
            date: d.date,
            revenue: d.value,
            expenses: dailyExp[d.date] || 0,
            profit: d.value - (dailyExp[d.date] || 0)
        }));
    }

    /**
     * Seasonal Demand Prediction.
     * Checks monthly trends and predicts demand for specific months.
     */
    async getSeasonalTrends() {
        const sales = await db.sales.toArray();
        const monthlyData = Array(12).fill(0).map((_, i) => ({ month: i, total: 0 }));

        sales.forEach(sale => {
            const date = new Date(sale.date);
            const month = date.getMonth();
            monthlyData[month].total += Number(sale.total || sale.grand_total || 0);
        });

        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const maxVal = Math.max(...monthlyData.map(m => m.total), 1);

        return monthlyData.map((m, i) => ({
            month: months[i],
            revenue: m.total,
            isPeak: m.total > 0 && m.total === maxVal
        }));
    }

    /**
     * Customer Churn Prediction (RFM Analysis).
     * Identifies customers at risk of leaving based on recency and frequency.
     */
    async getCustomerChurnInsight() {
        const customers = await db.customers.toArray();
        const sales = await db.sales.toArray();
        const now = new Date();

        const customerSales = {};
        sales.forEach(sale => {
            if (!sale.customer_id) return;
            const cid = sale.customer_id;
            if (!customerSales[cid]) {
                customerSales[cid] = { count: 0, lastDate: new Date(0), totalAmount: 0 };
            }
            customerSales[cid].count++;
            const saleDate = new Date(sale.date);
            if (saleDate > customerSales[cid].lastDate) {
                customerSales[cid].lastDate = saleDate;
            }
            customerSales[cid].totalAmount += Number(sale.total || sale.grand_total || 0);
        });

        return customers.map(c => {
            const stats = customerSales[c.id];
            if (!stats) return { ...c, risk: 'High', daysSinceLast: 'Never', color: '#f97316' };

            const daysSinceLast = Math.floor((now - stats.lastDate) / (1000 * 60 * 60 * 24));
            let risk = 'Low';
            if (daysSinceLast > 60) risk = 'Critical';
            else if (daysSinceLast > 30) risk = 'High';
            else if (daysSinceLast > 14) risk = 'Medium';

            return {
                id: c.id,
                name: c.name,
                risk,
                daysSinceLast,
                totalSpent: stats.totalAmount,
                orderCount: stats.count,
                color: risk === 'Critical' ? '#ef4444' : risk === 'High' ? '#f97316' : risk === 'Medium' ? '#eab308' : '#10b981'
            };
        }).sort((a, b) => {
            const order = { 'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3 };
            return order[a.risk] - order[b.risk];
        });
    }

    /**
     * AI-based Stock Ordering Suggestions.
     */
    async getLowStockOrders() {
        const products = await db.products.toArray();
        const lowStock = products.filter(p => (Number(p.stock_quantity) || 0) <= (Number(p.alert_quantity) || 5));

        return lowStock.map(p => ({
            id: p.id,
            name: p.name,
            currentStock: p.stock_quantity || 0,
            alertQty: p.alert_quantity || 5,
            suggestedOrder: Math.max(0, (p.alert_quantity || 5) * 2 - (p.stock_quantity || 0)),
            cost: p.cost || 0
        }));
    }
}

export const forecastingService = new ForecastingService();
