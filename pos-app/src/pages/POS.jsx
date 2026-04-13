import React, { useState, useEffect } from 'react';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';
import ProductGrid from '../components/pos/ProductGrid';
import Cart from '../components/pos/Cart';
import CheckoutModal from '../components/pos/CheckoutModal';
import RegisterShift from '../components/pos/RegisterShift';
import SuspendedSales from '../components/pos/SuspendedSales';
import ManualItemModal from '../components/pos/ManualItemModal';
import CustomerModal from '../components/pos/CustomerModal';
import ReturnsModal from '../components/pos/ReturnsModal';
import QuickExpenseModal from '../components/pos/QuickExpenseModal';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { triggerAutoBackup } from '../services/AutoBackupService';
import { FaPause, FaList, FaTimes, FaMoneyBillWave, FaUndo, FaWhatsapp } from 'react-icons/fa';
import { playBeep, playSuccess } from '../services/SoundService';
import { FaKeyboard } from 'react-icons/fa';

const POS = () => {
    const [error, setError] = useState(null);

    // Hooks must be called unconditionally - they will throw if context is not available
    const settingsContext = useSettings();
    const authContext = useAuth();

    // Safely extract values with defaults
    const settings = settingsContext?.settings || {};
    const formatPrice = settingsContext?.formatPrice || ((amount) => `Rs.${(amount || 0).toFixed(2)}`);
    const hasPermission = authContext?.hasPermission || (() => true);

    const products = useLiveQuery(() => db.products.toArray().catch(err => {
        console.error('Products query error:', err);
        setError('Database error: ' + err.message);
        return [];
    }));
    const customers = useLiveQuery(() => db.customers.toArray().catch(err => {
        console.error('Customers query error:', err);
        return [];
    }));
    const categories = useLiveQuery(() => db.categories.toArray().catch(err => {
        console.error('Categories query error:', err);
        return [];
    }));
    const brands = useLiveQuery(() => db.brands.toArray().catch(err => {
        console.error('Brands query error:', err);
        return [];
    }));

    // Check for open register to lock/unlock POS
    const openRegisters = useLiveQuery(() => db.registers.where('status').equals('open').toArray());
    const openRegister = openRegisters?.[0];

    const [cartItems, setCartItems] = useState([]);
    const [showCheckout, setShowCheckout] = useState(false);
    const [showSuspended, setShowSuspended] = useState(false);
    const [showManualItem, setShowManualItem] = useState(false);
    const [showQuickExpense, setShowQuickExpense] = useState(false);
    const [showCustomerModal, setShowCustomerModal] = useState(false);
    const [showReturns, setShowReturns] = useState(false);
    const [checkoutData, setCheckoutData] = useState({}); // Stores totals passed from Cart
    const [completedSale, setCompletedSale] = useState(null); // Stores sale data to show receipt after payment

    const [customer, setCustomer] = useState(null);

    const handleAddCustomer = async (data) => {
        const id = await db.customers.add(data);
        const newCustomer = { ...data, id };
        setCustomer(newCustomer);
        setShowCustomerModal(false);
        triggerAutoBackup(); // Auto-backup after adding customer
    };

    // Keyboard Shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                if (e.key === 'Escape') e.target.blur();
                return;
            }

            switch (e.key) {
                case 'F2':
                    e.preventDefault();
                    document.querySelector('input[placeholder*="Search"]')?.focus();
                    break;
                case 'F4':
                    e.preventDefault();
                    if (!showCheckout) setShowCustomerModal(true);
                    break;
                case 'F8':
                    e.preventDefault();
                    if (cartItems.length > 0) handleHold();
                    break;
                case 'F9':
                    e.preventDefault();
                    if (cartItems.length > 0 && !showCheckout) document.getElementById('btn-checkout')?.click();
                    break;
                case 'Escape':
                    setShowManualItem(false);
                    setShowCustomerModal(false);
                    setShowQuickExpense(false);
                    setShowReturns(false);
                    setShowSuspended(false);
                    setShowCheckout(false);
                    break;
                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [cartItems, showCheckout]);

    const warehouseStock = useLiveQuery(async () => {
        const whId = parseInt(settings?.default_warehouse || 1);
        return await db.stock.where('warehouse_id').equals(whId).toArray();
    }, [settings?.default_warehouse]);

    // Financial snapshot - MUST be called before any conditional returns
    const financialSnapshot = useLiveQuery(async () => {
        const sales = await db.sales.toArray();
        const expenses = await db.expenses.toArray();
        const returns = await db.returns.filter(r => r.type === 'sale').toArray();

        const totalSales = sales.reduce((sum, sale) => sum + (sale.grandTotal || sale.total || 0), 0);
        const totalExpenses = expenses.reduce((sum, exp) => sum + (parseFloat(exp.amount) || 0), 0);
        const totalReturns = returns.reduce((sum, ret) => sum + (ret.grandTotal || ret.total || ret.totalRefund || 0), 0);

        return { totalSales, totalExpenses, totalReturns };
    }, []);

    const { totalSales = 0, totalExpenses = 0, totalReturns = 0 } = financialSnapshot || {};

    // Show loading only if data is truly not available (not just empty arrays)
    // ALL HOOKS MUST BE CALLED BEFORE THIS POINT
    if (products === undefined || warehouseStock === undefined || categories === undefined || brands === undefined || openRegisters === undefined) {
        return <div style={{ padding: '20px', textAlign: 'center' }}>Loading POS Data...</div>;
    }

    const getStockForProduct = (productId) => {
        // 1. Find Warehouse-Specific Stock
        const s = warehouseStock?.find(s => s.product_id === productId);
        
        // 2. If a warehouse record exists, return that specific quantity (even if 0)
        if (s !== undefined && s !== null) {
            return parseFloat(s.quantity || 0);
        }

        // 3. Fallback to Global Stock only if no warehouse record exists (initial setup)
        const p = products?.find(p => p.id === productId);
        return parseFloat(p?.stock_quantity || 0);
    };

    const addToCart = (product) => {
        playBeep();
        const available = getStockForProduct(product.id);
        const existingItem = cartItems.find(item => item.id === product.id);
        const currentQty = existingItem ? existingItem.quantity : 0;

        if (currentQty + 1 > available) {
            // Debugging Info for User
            const s = warehouseStock?.find(s => s.product_id === product.id);
            const p = products?.find(p => p.id === product.id);
            const debugInfo = `(WH: ${s ? s.quantity : 'N/A'}, GL: ${p ? p.stock_quantity : 'N/A'})`;

            return alert(`Insufficient stock! Only ${available} available. ${debugInfo}`);
        }

        if (existingItem) {
            setCartItems(cartItems.map(item =>
                item.id === product.id ? { ...item, quantity: item.quantity + 1 } : item
            ));
        } else {
            setCartItems([...cartItems, { ...product, quantity: 1 }]);
        }
    };

    const removeFromCart = (id) => {
        setCartItems(cartItems.filter(item => item.id !== id));
    };

    const updateQuantity = (id, delta, absolute = false) => {
        const item = cartItems.find(i => i.id === id);
        if (!item) return;

        const newQty = absolute ? delta : item.quantity + delta;
        if (newQty < 1) return;

        // Skip stock check for manual items
        if (item.isManual) {
            setCartItems(cartItems.map(i =>
                i.id === id ? { ...i, quantity: newQty } : i
            ));
            return;
        }

        const available = getStockForProduct(id);
        if (newQty > available) {
            alert(`Insufficient stock! Max available: ${available}`);
            return;
        }

        setCartItems(cartItems.map(i =>
            i.id === id ? { ...i, quantity: newQty } : i
        ));
    };

    const clearCart = () => setCartItems([]);

    const handleCheckoutInit = (cartSummary) => {
        setCheckoutData(cartSummary); // Store summary in state to pass to modal
        setShowCheckout(true);
    };

    const handleCheckoutComplete = async (paymentDetails) => {
        const whId = parseInt(settings.default_warehouse || 1);

        // Final Stock Validation
        for (const item of cartItems) {
            if (item.isManual) continue;
            const available = getStockForProduct(item.id);
            if (item.quantity > available) {
                const s = warehouseStock?.find(s => s.product_id === item.id);
                const p = products?.find(p => p.id === item.id);
                const debugInfo = `(WH: ${s ? s.quantity : 'N/A'}, GL: ${p ? p.stock_quantity : 'N/A'})`;

                return alert(`Insufficient stock for "${item.name}"! Available: ${available}. ${debugInfo} Please adjust your cart.`);
            }
        }

        const sale = {
            date: new Date(),
            items: cartItems.map(i => ({ ...i, quantity: i.quantity, price: i.price, cost: i.cost })), // Snapshot
            ...checkoutData, // subtotal, tax, discount, total, etc.
            warehouse_id: whId,
            paymentMethod: paymentDetails.method, // "Split" or "Cash" etc.
            payments: paymentDetails.payments || [], // Array of split payments
            status: 'completed',
            customer_id: customer?.id || null
        };

        const saleId = await db.sales.add(sale);
        const finalSale = { ...sale, id: saleId };

        // Handle Delivery Registration
        if (paymentDetails.isDelivery) {
            await db.deliveries.add({
                sale_id: saleId,
                date: new Date(),
                status: 'packing',
                customer_id: customer?.id || null,
                address: paymentDetails.shippingDetails?.address || '',
                note: paymentDetails.shippingDetails?.notes || '',
                reference_no: '' // To be assigned by warehouse
            });
        }

        // Add split payments info to db if needed
        if (paymentDetails.payments) {
            await db.sale_payments.bulkAdd(paymentDetails.payments.map(p => ({
                sale_id: saleId,
                date: new Date(),
                method: p.method,
                amount: parseFloat(p.amount)
            })));
        }

        // Update stock
        for (const item of cartItems) {
            // 1. Fetch current states
            const existingStock = await db.stock
                .where('product_id').equals(item.id)
                .filter(s => s.warehouse_id === whId)
                .first();

            const product = await db.products.get(item.id);
            const globalQty = product?.stock_quantity || 0;
            const currentWhQty = existingStock ? (existingStock.quantity || 0) : 0;

            // 2. Determine if we need to Adopt Global Stock (Self-Healing)
            // If WhStock is insufficient but Global is sufficient, use Global as base.
            let quantityBase = currentWhQty;
            let stockId = existingStock?.id;

            if (currentWhQty < item.quantity && globalQty >= item.quantity) {
                // HEALING: Adopt global quantity into this warehouse
                quantityBase = globalQty;
            }

            // 3. Apply Update
            const newQty = quantityBase - item.quantity;

            if (stockId) {
                await db.stock.update(stockId, { quantity: newQty });
            } else {
                await db.stock.add({
                    product_id: item.id,
                    variant_id: 0,
                    warehouse_id: whId,
                    quantity: newQty
                });
            }

            // 4. Global Stock Re-sync
            const allWhStocks = await db.stock.where('product_id').equals(item.id).toArray();
            const totalWhStock = allWhStocks.reduce((sum, s) => sum + s.quantity, 0);
            await db.products.update(item.id, { stock_quantity: totalWhStock });
        }

        setShowCheckout(false);
        setCartItems([]);
        setCompletedSale(finalSale);
        setCompletedSale(finalSale);
        playSuccess();
        triggerAutoBackup(); // Auto-backup after completing sale
    };

    const handleHold = async () => {
        if (cartItems.length === 0) return alert('Cart is empty');
        const hasInvalidQty = cartItems.some(item => item.quantity <= 0);
        if (hasInvalidQty) return alert('All items must have a quantity greater than 0.');

        const total = cartItems.reduce((acc, item) => acc + (item.price * item.quantity), 0);
        await db.suspended_sales.add({
            date: new Date(),
            items: cartItems,
            customer_id: customer?.id,
            customer_name: customer?.name,
            total: total
        });
        setCartItems([]);
        alert('Sale Suspended!');
        triggerAutoBackup(); // Auto-backup after suspending sale
    };

    const handleResume = (suspendedCart) => {
        setCartItems(suspendedCart.items);
        setShowSuspended(false);
    };

    // Show error if context failed
    if (error) {
        return (
            <div style={{ padding: '40px', textAlign: 'center', background: '#fff1f2', borderRadius: '12px', margin: '20px' }}>
                <h2 style={{ color: '#e11d48', marginBottom: '10px' }}>Error Loading POS</h2>
                <p style={{ color: '#991b1b', marginBottom: '20px' }}>{error}</p>
                <button className="btn btn-primary" onClick={() => window.location.reload()}>Refresh Page</button>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '15px' }}>
            <RegisterShift />

            {openRegister && (
                <div className="pos-container" style={{ display: 'flex', flex: 1, gap: '20px', minHeight: 0, position: 'relative' }}>
                    {/* Shortcuts Legend */}
                    <div style={{ position: 'absolute', bottom: '10px', left: '10px', fontSize: '0.75rem', color: '#94a3b8', display: 'flex', gap: '12px', zIndex: 10, background: 'rgba(255,255,255,0.8)', padding: '4px 8px', borderRadius: '4px' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><FaKeyboard /> Shortcuts:</span>
                        <span><b>F2</b> Search</span>
                        <span><b>F4</b> Customer</span>
                        <span><b>F8</b> Hold</span>
                        <span><b>F9</b> Pay</span>
                    </div>
                    {showSuspended && <SuspendedSales onResume={handleResume} onClose={() => setShowSuspended(false)} />}
                    {showManualItem && <ManualItemModal onClose={() => setShowManualItem(false)} onAdd={(item, qty) => { setCartItems([...cartItems, { ...item, quantity: qty }]); setShowManualItem(false); }} />}
                    {showQuickExpense && <QuickExpenseModal onClose={() => setShowQuickExpense(false)} />}
                    {showCustomerModal && <CustomerModal onClose={() => setShowCustomerModal(false)} onAdd={handleAddCustomer} />}
                    {showReturns && <ReturnsModal onClose={() => setShowReturns(false)} />}

                    <div className="pos-left" style={{ flex: '1 1 65%', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
                            <div style={{ display: 'flex', alignItems: 'baseline', gap: '15px', flexWrap: 'wrap' }}>
                                <h2 style={{ fontSize: '1.5rem', fontWeight: '800' }}>POS Terminal</h2>
                                <div style={{ background: '#eee', padding: '2px 8px', borderRadius: '4px', fontSize: '12px', color: '#555' }}>
                                    Warehouse: {settings.default_warehouse ? 'Main' : 'Default'}
                                </div>
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                    <div style={{
                                        background: 'rgba(16, 185, 129, 0.1)',
                                        padding: '4px 12px',
                                        borderRadius: '20px',
                                        fontSize: '0.8rem',
                                        fontWeight: '600',
                                        color: '#059669',
                                        border: '1px solid rgba(16, 185, 129, 0.2)'
                                    }}>
                                        Sales: <span style={{ fontWeight: '800' }}>{formatPrice(totalSales)}</span>
                                    </div>
                                    <div style={{
                                        background: 'rgba(244, 63, 94, 0.1)',
                                        padding: '4px 12px',
                                        borderRadius: '20px',
                                        fontSize: '0.8rem',
                                        fontWeight: '600',
                                        color: '#e11d48',
                                        border: '1px solid rgba(244, 63, 94, 0.2)'
                                    }}>
                                        Expenses: <span style={{ fontWeight: '800' }}>{formatPrice(totalExpenses)}</span>
                                    </div>
                                    <div style={{
                                        background: 'rgba(245, 158, 11, 0.1)',
                                        padding: '4px 12px',
                                        borderRadius: '20px',
                                        fontSize: '0.8rem',
                                        fontWeight: '600',
                                        color: '#d97706',
                                        border: '1px solid rgba(245, 158, 11, 0.2)'
                                    }}>
                                        Returns: <span style={{ fontWeight: '800' }}>{formatPrice(totalReturns)}</span>
                                    </div>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                    <select className="form-control" style={{ width: '180px' }} value={customer?.id || ''} onChange={e => {
                                        const c = (customers || []).find(x => x.id === parseInt(e.target.value));
                                        setCustomer(c || null);
                                    }}>
                                        <option value="">Walk-in Customer</option>
                                        {(customers || []).map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                                    </select>
                                    <button className="btn btn-primary" style={{ padding: '8px 12px' }} onClick={() => setShowCustomerModal(true)} title="Add Customer">+</button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {hasPermission('pos_manual_item') && <button className="btn btn-secondary" onClick={() => setShowManualItem(true)}><FaList /> Manual</button>}
                                    {hasPermission('pos_hold_sale') && <button className="btn btn-warning" onClick={handleHold}><FaPause /> Hold</button>}
                                    {hasPermission('pos_hold_sale') && <button className="btn btn-info" onClick={() => setShowSuspended(true)}><FaList /> Lists</button>}
                                    {hasPermission('pos_returns') && <button className="btn btn-accent" onClick={() => setShowReturns(true)} title="Process Return" style={{ background: 'var(--accent-color)', color: 'white' }}><FaUndo /> Return</button>}
                                    {hasPermission('manage_settings') && <button className="btn btn-danger" onClick={() => setShowQuickExpense(true)} title="Quick Expense"><FaMoneyBillWave /></button>}
                                </div>
                            </div>
                        </div>
                        <div style={{ flex: 1, overflowY: 'auto', paddingRight: '10px' }}>
                            <ProductGrid
                                products={products}
                                categories={categories}
                                brands={brands}
                                onAddToCart={addToCart}
                                getStock={getStockForProduct}
                            />
                        </div>
                    </div>

                    <div className="pos-right" style={{ flex: '0 0 420px', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                        <Cart
                            cart={cartItems}
                            onRemove={removeFromCart}
                            onUpdateQuantity={updateQuantity}
                            onPriceUpdate={(id, newPrice) => setCartItems(cartItems.map(item => item.id === id ? { ...item, price: newPrice } : item))}
                            onClear={clearCart}
                            onCheckout={handleCheckoutInit}
                        />
                    </div>
                </div>
            )}

            {!openRegister && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc', borderRadius: '12px', border: '2px dashed #cbd5e1' }}>
                    <div style={{ textAlign: 'center', color: '#64748b' }}>
                        <h2 style={{ fontSize: '1.5rem', marginBottom: '10px' }}>🔒 POS Locked</h2>
                        <p>Please open the register to begin processing sales.</p>
                    </div>
                </div>
            )}

            {showCheckout && (
                <CheckoutModal
                    total={checkoutData.total}
                    onConfirm={handleCheckoutComplete}
                    onCancel={() => setShowCheckout(false)}
                />
            )}

            {completedSale && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1100
                }}>
                    <div className="modal-content card" style={{ padding: '25px', width: '400px', background: 'white', textAlign: 'center' }}>
                        <div id="pos-receipt-print" style={{ textAlign: 'center', color: '#000', fontSize: '14px', fontFamily: 'monospace' }}>
                            <h2 style={{ margin: '0 0 5px 0', textTransform: 'uppercase' }}>{settings.site_name || 'RECEIPT'}</h2>
                            {settings.pos_receipt_header && <p style={{ margin: '0 0 10px 0' }}>{settings.pos_receipt_header}</p>}
                            <div style={{ borderTop: '1px dashed #000', borderBottom: '1px dashed #000', padding: '10px 0', margin: '10px 0', textAlign: 'left' }}>
                                <p style={{ margin: '2px 0' }}>Sale ID: #{completedSale.id}</p>
                                <p style={{ margin: '2px 0' }}>Date: {new Date(completedSale.date).toLocaleString()}</p>
                                <p style={{ margin: '2px 0' }}>Customer: {customer?.name || 'Walk-in'}</p>
                            </div>
                            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', marginBottom: '10px' }}>
                                <tbody>
                                    {completedSale.items.map((item, idx) => (
                                        <tr key={idx}>
                                            <td style={{ padding: '2px 0' }}>{item.name} x{item.quantity}</td>
                                            <td style={{ textAlign: 'right', padding: '2px 0' }}>{formatPrice(item.price * item.quantity)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            <div style={{ borderTop: '1px dashed #000', paddingTop: '10px', textAlign: 'right' }}>
                                <p style={{ margin: '2px 0' }}>Subtotal: {formatPrice(completedSale.subtotal)}</p>
                                {completedSale.tax > 0 && <p style={{ margin: '2px 0' }}>Tax: {formatPrice(completedSale.tax)}</p>}
                                {completedSale.discount > 0 && <p style={{ margin: '2px 0' }}>Discount: -{formatPrice(completedSale.discount)}</p>}
                                <h3 style={{ margin: '5px 0' }}>Total: {formatPrice(completedSale.total)}</h3>
                                <p style={{ margin: '2px 0' }}>Method: {completedSale.paymentMethod}</p>
                            </div>
                            {settings.pos_receipt_footer && (
                                <div style={{ marginTop: '20px', borderTop: '1px dashed #000', paddingTop: '10px' }}>
                                    <p>{settings.pos_receipt_footer}</p>
                                </div>
                            )}
                        </div>
                        <div className="no-print" style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                            <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => { window.print(); setCompletedSale(null); }}>Print & New</button>
                            <button className="btn" style={{ background: '#25D366', color: 'white', flex: 1, border: 'none' }} onClick={() => {
                                const text = `*RECEIPT*\n${settings.site_name || 'Store'}\nDate: ${new Date(completedSale.date).toLocaleString()}\nSale ID: #${completedSale.id}\n----------------\n${completedSale.items.map(i => `${i.name} x${i.quantity} ${formatPrice(i.price * i.quantity)}`).join('\n')}\n----------------\n*Total: ${formatPrice(completedSale.total)}*\nThank you!`;
                                const phone = customer?.phone ? customer.phone.replace(/\D/g, '') : '';
                                window.open(`https://wa.me/${phone}?text=${encodeURIComponent(text)}`, '_blank');
                            }}><FaWhatsapp /> WhatsApp</button>
                            <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setCompletedSale(null)}>Close</button>
                        </div>
                    </div>
                    <style>{`
                        @media print {
                            body * { visibility: hidden; }
                            #pos-receipt-print, #pos-receipt-print * { visibility: visible; }
                            #pos-receipt-print { position: fixed; left: 0; top: 0; width: 100%; padding: 20px; }
                            .no-print { display: none; }
                        }
                    `}</style>
                </div>
            )}
        </div>
    );
};

export default POS;
