import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';
import { FaPlus, FaTrash, FaEye, FaPrint, FaSearch, FaEdit } from 'react-icons/fa';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';

const Purchases = () => {
    const { formatPrice } = useSettings();
    const { hasPermission } = useAuth();
    const purchases = useLiveQuery(() => db.purchases.orderBy('date').reverse().toArray());
    const suppliers = useLiveQuery(() => db.suppliers.toArray());
    const products = useLiveQuery(() => db.products.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [view, setView] = useState('list'); // 'list' | 'form' | 'edit'
    const [selectedPurchase, setSelectedPurchase] = useState(null);

    // Filters
    const [searchTerm, setSearchTerm] = useState('');
    const [filterSupplier, setFilterSupplier] = useState('');
    const [filterWarehouse, setFilterWarehouse] = useState('');
    const [dateRange, setDateRange] = useState({ start: '', end: '' });

    // Form State
    const [cart, setCart] = useState([]);
    const [formData, setFormData] = useState({
        supplier_id: '',
        warehouse_id: '',
        reference_no: '',
        status: 'received',
        order_tax: 0,
        shipping_cost: 0,
        paid_amount: 0,
        payment_method: 'Cash',
        notes: ''
    });

    const addToCart = (productId) => {
        const product = products.find(p => p.id === parseInt(productId));
        if (!product) return;

        const existing = cart.find(item => item.id === product.id);
        if (existing) {
            setCart(cart.map(item => item.id === product.id ? { ...item, 
                : item.quantity + 1 } : item));
        } else {
            setCart([...cart, { ...product, quantity: 1, cost: product.cost }]);
        }
    };

    const updateCartItem = (id, field, value) => {
        setCart(cart.map(item => item.id === id ? { ...item, [field]: value } : item));
    };

    const removeCartItem = (id) => setCart(cart.filter(item => item.id !== id));

    const handleEditClick = (p) => {
        setSelectedPurchase(p);
        setFormData({
            supplier_id: p.supplier_id || '',
            warehouse_id: p.warehouse_id || '',
            reference_no: p.reference_no || '',
            status: p.status || 'received',
            order_tax: p.order_tax || 0,
            shipping_cost: p.shipping || 0,
            paid_amount: p.paid_amount || 0,
            payment_method: p.payment_method || 'Cash',
            notes: p.notes || ''
        });
        setCart(p.items || []);
        setView('edit');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (cart.length === 0) return alert('Please add products to the purchase');

        const subtotal = cart.reduce((sum, item) => sum + (item.cost * item.quantity), 0);
        const taxAmount = (subtotal * (parseFloat(formData.order_tax) || 0)) / 100;
        const grandTotal = subtotal + taxAmount + (parseFloat(formData.shipping_cost) || 0);

        const purchaseRecord = {
            ...formData,
            supplier_id: parseInt(formData.supplier_id),
            warehouse_id: parseInt(formData.warehouse_id),
            order_tax: parseFloat(formData.order_tax) || 0, // Persist percentage for editing
            subtotal: subtotal,
            tax: taxAmount,
            shipping: parseFloat(formData.shipping_cost) || 0,
            grand_total: grandTotal,
            paid_amount: parseFloat(formData.paid_amount) || 0,
            items: cart,
            payment_status: parseFloat(formData.paid_amount) >= grandTotal ? 'paid' : (parseFloat(formData.paid_amount) > 0 ? 'partial' : 'pending'),
        };

        if (view === 'edit' && selectedPurchase) {
            // 1. REVERT OLD STOCK if it was received
            if (selectedPurchase.status === 'received') {
                const oldWhId = parseInt(selectedPurchase.warehouse_id);
                for (const item of selectedPurchase.items) {
                    const existingStock = await db.stock
                        .where('product_id').equals(item.id)
                        .filter(s => s.warehouse_id === oldWhId)
                        .first();
                    if (existingStock) {
                        await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) - item.quantity });
                    }
                    const product = await db.products.get(item.id);
                    if (product) {
                        await db.products.update(item.id, { stock_quantity: (product.stock_quantity || 0) - item.quantity });
                    }
                }
            }

            // 2. UPDATE PURCHASE
            await db.purchases.update(selectedPurchase.id, {
                ...purchaseRecord,
                date: selectedPurchase.date,
                source: selectedPurchase.source || 'manual'
            });

            // 3. APPLY NEW STOCK if new status is received
            if (formData.status === 'received') {
                const newWhId = parseInt(formData.warehouse_id);
                for (const item of cart) {
                    const existingStock = await db.stock
                        .where('product_id').equals(item.id)
                        .filter(s => s.warehouse_id === newWhId)
                        .first();

                    if (existingStock) {
                        await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) + parseInt(item.quantity) });
                    } else {
                        await db.stock.add({ product_id: item.id, variant_id: 0, warehouse_id: newWhId, quantity: parseInt(item.quantity) });
                    }

                    const product = await db.products.get(item.id);
                    if (product) {
                        await db.products.update(item.id, {
                            stock_quantity: (product.stock_quantity || 0) + parseInt(item.quantity),
                            cost: parseFloat(item.cost) || product.cost
                        });
                    }
                }
            }
            alert('Purchase updated successfully!');
        } else {
            // CREATE NEW
            await db.purchases.add({
                ...purchaseRecord,
                date: new Date(),
                source: 'manual'
            });

            // Update Stock if Received
            if (formData.status === 'received') {
                const whId = parseInt(formData.warehouse_id);
                for (const item of cart) {
                    const existingStock = await db.stock
                        .where('product_id').equals(item.id)
                        .filter(s => s.warehouse_id === whId)
                        .first();

                    if (existingStock) {
                        await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) + parseInt(item.quantity) });
                    } else {
                        await db.stock.add({ product_id: item.id, variant_id: 0, warehouse_id: whId, quantity: parseInt(item.quantity) });
                    }

                    const product = await db.products.get(item.id);
                    if (product) {
                        await db.products.update(item.id, {
                            stock_quantity: (product.stock_quantity || 0) + parseInt(item.quantity),
                            cost: parseFloat(item.cost) || product.cost
                        });
                    }
                }
            }
            alert('Purchase added successfully!');
        }

        setFormData({
            supplier_id: '', warehouse_id: '', reference_no: '', status: 'received',
            order_tax: 0, shipping_cost: 0, paid_amount: 0, payment_method: 'Cash', notes: ''
        });
        setCart([]);
        setView('list');
        setSelectedPurchase(null);
    };

    // CSV Import
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (evt) => {
            const text = evt.target.result;
            const rows = text.split('\n').slice(1);

            let count = 0;
            for (const row of rows) {
                if (!row.trim()) continue;
                // Date,Ref,Supplier,Total,Status
                const [dateStr, ref, supplierName, totalStr, status] = row.split(',');

                const supplier = await db.suppliers.where('name').equalsIgnoreCase(supplierName?.trim() || '').first();

                await db.purchases.add({
                    date: new Date(dateStr || new Date()),
                    reference_no: ref || `CSV-${Date.now()}`,
                    supplier_id: supplier?.id || null,
                    warehouse_id: null,
                    grand_total: parseFloat(totalStr) || 0,
                    status: status?.trim().toLowerCase() || 'received',
                    items: [],
                    source: 'csv_import'
                });
                count++;
            }
            alert(`Imported ${count} purchases!`);
            e.target.value = '';
        };
        reader.readAsText(file);
    };

    const handleDelete = async (purchase) => {
        if (!window.confirm('Delete this purchase? Stock will be REVERTED if status was "Received".')) return;

        // Revert Stock
        if (purchase.status === 'received') {
            const whId = parseInt(purchase.warehouse_id);
            for (const item of purchase.items) {
                // 1. Revert Warehouse Stock
                const existingStock = await db.stock
                    .where('product_id').equals(item.id)
                    .filter(s => s.warehouse_id === whId)
                    .first();
                if (existingStock) {
                    await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) - item.quantity });
                }

                // 2. Revert Global Stock
                const product = await db.products.get(item.id);
                if (product) {
                    await db.products.update(item.id, { stock_quantity: (product.stock_quantity || 0) - item.quantity });
                }
            }
        }
        await db.purchases.delete(purchase.id);
    };

    // Filter Logic
    const filteredPurchases = purchases?.filter(p => {
        const pDate = new Date(p.date);
        const start = dateRange.start ? new Date(dateRange.start) : null;
        const end = dateRange.end ? new Date(dateRange.end) : null;
        if (end) end.setHours(23, 59, 59);

        const matchesSearch = searchTerm === '' ||
            (p.reference_no?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                p.id.toString().includes(searchTerm));

        const matchesSupplier = filterSupplier ? p.supplier_id === parseInt(filterSupplier) : true;
        const matchesWarehouse = filterWarehouse ? p.warehouse_id === parseInt(filterWarehouse) : true;
        const matchesDate = (!start || pDate >= start) && (!end || pDate <= end);

        return matchesSearch && matchesSupplier && matchesWarehouse && matchesDate;
    });

    // Calculates Form Totals for Display
    const currentSubtotal = cart.reduce((sum, item) => sum + (item.cost * item.quantity), 0);
    const currentTax = (currentSubtotal * (parseFloat(formData.order_tax) || 0)) / 100;
    const currentTotal = currentSubtotal + currentTax + (parseFloat(formData.shipping_cost) || 0);


    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Purchases</h2>
                {view === 'list' ?
                    (hasPermission('purchases_add') && <button className="btn btn-primary" onClick={() => { setView('form'); setSelectedPurchase(null); setFormData({ supplier_id: '', warehouse_id: '', reference_no: '', status: 'received', order_tax: 0, shipping_cost: 0, paid_amount: 0, payment_method: 'Cash', notes: '' }); setCart([]); }}><FaPlus /> Add Purchase</button>) :
                    <button className="btn btn-secondary" onClick={() => setView('list')}>Back to List</button>
                }
            </div>

            {view === 'list' && (
                <div className="card">
                    {/* Filters */}
                    <div className="mb-3 p-3 bg-light rounded" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                        <div>
                            <label>Search</label>
                            <input className="form-control" placeholder="Ref No or ID..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                        </div>
                        <div>
                            <label>Supplier</label>
                            <select className="form-control" value={filterSupplier} onChange={e => setFilterSupplier(e.target.value)}>
                                <option value="">All Suppliers</option>
                                {suppliers?.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                            </select>
                        </div>
                        <div>
                            <label>Warehouse</label>
                            <select className="form-control" value={filterWarehouse} onChange={e => setFilterWarehouse(e.target.value)}>
                                <option value="">All Warehouses</option>
                                {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                            </select>
                        </div>
                        <div>
                            <label>Date Range</label>
                            <div style={{ display: 'flex', gap: '5px' }}>
                                <input type="date" className="form-control" value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
                                <span style={{ alignSelf: 'center' }}>-</span>
                                <input type="date" className="form-control" value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
                            </div>
                        </div>

                        {hasPermission('purchases_add') && (
                            <div style={{ marginLeft: 'auto', display: 'flex', flexDirection: 'column' }}>
                                <label>Import Purchases (CSV)</label>
                                <input type="file" accept=".csv" onChange={handleFileUpload} className="form-control" style={{ width: '250px' }} />
                            </div>
                        )}
                    </div>

                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                                <th style={{ padding: '10px' }}>Date</th>
                                <th style={{ padding: '10px' }}>Ref</th>
                                <th style={{ padding: '10px' }}>Supplier</th>
                                <th style={{ padding: '10px' }}>Warehouse</th>
                                <th style={{ padding: '10px' }}>Status</th>
                                <th style={{ padding: '10px' }}>Total</th>
                                <th style={{ padding: '10px' }}>Paid</th>
                                <th style={{ padding: '10px' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredPurchases?.map(p => (
                                <tr key={p.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                    <td style={{ padding: '10px' }}>{new Date(p.date).toLocaleDateString()}</td>
                                    <td style={{ padding: '10px' }}>{p.reference_no}</td>
                                    <td style={{ padding: '10px' }}>{suppliers?.find(s => s.id === p.supplier_id)?.name || 'N/A'}</td>
                                    <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id === p.warehouse_id)?.name || 'Default'}</td>
                                    <td style={{ padding: '10px' }}>
                                        <span style={{ padding: '4px 8px', borderRadius: '4px', background: p.status === 'received' ? '#d4edda' : '#fff3cd' }}>
                                            {p.status}
                                        </span>
                                    </td>
                                    <td style={{ padding: '10px' }}>{formatPrice(p.grand_total)}</td>
                                    <td style={{ padding: '10px' }}>{formatPrice(p.paid_amount || 0)}</td>
                                    <td style={{ padding: '10px', display: 'flex', gap: '5px' }}>
                                        <button className="btn-icon" onClick={() => { setSelectedPurchase(p); setView('list'); }} title="View Details"><FaEye /></button>
                                        <button className="btn-icon" style={{ color: 'var(--primary-color)' }} onClick={() => handleEditClick(p)} title="Edit Purchase"><FaEdit /></button>
                                        {hasPermission('purchases_delete') && (
                                            <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(p)} title="Delete Purchase"><FaTrash /></button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {filteredPurchases?.length === 0 && <tr><td colSpan="8" style={{ padding: '20px', textAlign: 'center' }}>No purchases found.</td></tr>}
                        </tbody>
                    </table>
                </div>
            )}

            {(view === 'form' || view === 'edit') && (
                <div className="card mb-3">
                    <h3>{view === 'edit' && selectedPurchase ? `Edit Purchase #${selectedPurchase.id}` : 'New Purchase'}</h3>
                    <form onSubmit={handleSubmit}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                            <div>
                                <label>Date</label>
                                <input type="text" className="form-control" value={new Date().toLocaleDateString()} disabled />
                            </div>
                            <div>
                                <label>Supplier</label>
                                <select className="form-control" required value={formData.supplier_id} onChange={e => setFormData({ ...formData, supplier_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                    <option value="">Select Supplier</option>
                                    {suppliers?.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                                </select>
                            </div>
                            <div>
                                <label>Warehouse</label>
                                <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                    <option value="">Select Warehouse</option>
                                    {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                                </select>
                            </div>
                            <div>
                                <label>Reference No</label>
                                <input type="text" className="form-control" value={formData.reference_no} onChange={e => setFormData({ ...formData, reference_no: e.target.value })} style={{ width: '100%', padding: '8px' }} />
                            </div>
                            <div>
                                <label>Status</label>
                                <select className="form-control" value={formData.status} onChange={e => setFormData({ ...formData, status: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                    <option value="received">Received</option>
                                    <option value="ordered">Ordered</option>
                                    <option value="pending">Pending</option>
                                </select>
                            </div>
                        </div>

                        <div style={{ marginBottom: '15px' }}>
                            <label>Add Product</label>
                            <select className="form-control" onChange={e => { addToCart(e.target.value); e.target.value = ''; }} style={{ width: '100%', padding: '8px' }}>
                                <option value="">Search/Select Product to add...</option>
                                {products?.map(p => <option key={p.id} value={p.id}>{p.name} ({p.code})</option>)}
                            </select>
                        </div>

                        <table style={{ width: '100%', marginBottom: '15px' }}>
                            <thead>
                                <tr style={{ background: '#f9f9f9', textAlign: 'left' }}>
                                    <th style={{ padding: '8px' }}>Product</th>
                                    <th style={{ padding: '8px' }}>Cost</th>
                                    <th style={{ padding: '8px' }}>Quantity</th>
                                    <th style={{ padding: '8px' }}>Subtotal</th>
                                    <th style={{ padding: '8px' }}></th>
                                </tr>
                            </thead>
                            <tbody>
                                {cart.map(item => (
                                    <tr key={item.id}>
                                        <td style={{ padding: '8px' }}>{item.name}</td>
                                        <td style={{ padding: '8px' }}>
                                            <input type="number" min="0" step="0.01" value={item.cost} onChange={e => updateCartItem(item.id, 'cost', parseFloat(e.target.value))} style={{ width: '80px' }} />
                                        </td>
                                        <td style={{ padding: '8px' }}>
                                            <input type="number" min="0.001" step="any" value={item.quantity} onChange={e => updateCartItem(item.id, 'quantity', parseFloat(e.target.value))} style={{ width: '60px' }} />
                                        </td>
                                        <td style={{ padding: '8px' }}>{formatPrice(item.cost * item.quantity)}</td>
                                        <td style={{ padding: '8px' }}>
                                            <button type="button" className="btn-icon" onClick={() => removeCartItem(item.id)} style={{ color: 'red' }}><FaTrash /></button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <div className="card bg-light p-3 mb-3">
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                <div>
                                    <label>Order Tax (%)</label>
                                    <input type="number" min="0" step="0.01" className="form-control" value={formData.order_tax} onChange={e => setFormData({ ...formData, order_tax: e.target.value })} />
                                </div>
                                <div>
                                    <label>Shipping Cost</label>
                                    <input type="number" min="0" step="0.01" className="form-control" value={formData.shipping_cost} onChange={e => setFormData({ ...formData, shipping_cost: e.target.value })} />
                                </div>
                            </div>
                        </div>

                        <div className="card p-3 mb-3" style={{ background: '#e8f5e9' }}>
                            <h4 style={{ marginTop: 0 }}>Values</h4>
                            <p>Subtotal: {formatPrice(currentSubtotal)}</p>
                            <p>Tax: {formatPrice(currentTax)}</p>
                            <p>Shipping: {formatPrice(parseFloat(formData.shipping_cost || 0))}</p>
                            <h3>Grand Total: {formatPrice(currentTotal)}</h3>
                        </div>

                        <div className="card bg-light p-3 mb-3">
                            <h4>Payment</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                <div>
                                    <label>Paid Amount</label>
                                    <input type="number" min="0" step="0.01" className="form-control" value={formData.paid_amount} onChange={e => setFormData({ ...formData, paid_amount: e.target.value })} />
                                </div>
                                <div>
                                    <label>Payment Method</label>
                                    <select className="form-control" value={formData.payment_method} onChange={e => setFormData({ ...formData, payment_method: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                        <option value="Cash">Cash</option>
                                        <option value="Card">Card</option>
                                        <option value="Cheque">Cheque</option>
                                        <option value="Bank Transfer">Bank Transfer</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginBottom: '15px' }}>
                            <label>Notes</label>
                            <textarea className="form-control" rows="3" value={formData.notes} onChange={e => setFormData({ ...formData, notes: e.target.value })}></textarea>
                        </div>

                        <button className="btn btn-primary" type="submit">{view === 'edit' ? 'Update Purchase' : 'Submit Purchase'}</button>
                    </form>
                </div>
            )}

            {/* Purchase Details Modal - Only show if in list view to avoid overlapping with Edit form */}
            {view === 'list' && selectedPurchase && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)',
                    display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
                }}>
                    <div className="modal-content card" style={{ width: '600px', maxHeight: '90vh', overflowY: 'auto', padding: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <h3>Purchase #{selectedPurchase.id}</h3>
                            <button className="btn" onClick={() => setSelectedPurchase(null)}>Close</button>
                        </div>

                        <div id="purchase-print-area" style={{ padding: '20px', border: '1px solid #eee', margin: '20px 0' }}>
                            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                                <h1>PURCHASE ORDER</h1>
                                <p>Date: {new Date(selectedPurchase.date).toLocaleDateString()}</p>
                                <p>Ref: {selectedPurchase.reference_no}</p>
                                <p>Supplier: {suppliers?.find(s => s.id === selectedPurchase.supplier_id)?.name}</p>
                            </div>
                            <table style={{ width: '100%', marginBottom: '20px' }}>
                                <thead><tr style={{ borderBottom: '1px solid #000' }}><th>Item</th><th>Qty</th><th>Cost</th><th>Total</th></tr></thead>
                                <tbody>
                                    {selectedPurchase.items.map((item, i) => (
                                        <tr key={i}>
                                            <td>{item.name}</td>
                                            <td>{item.quantity}</td>
                                            <td>{formatPrice(item.cost)}</td>
                                            <td>{formatPrice(item.cost * item.quantity)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            <div style={{ textAlign: 'right' }}>
                                <h3>Total: {formatPrice(selectedPurchase.grand_total)}</h3>
                            </div>
                            <div style={{ marginTop: '20px', fontSize: '12px' }}>
                                <strong>Notes:</strong>
                                <p>{selectedPurchase.notes}</p>
                            </div>
                        </div>

                        <div className="no-print" style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                            <button className="btn btn-primary" onClick={() => { handleEditClick(selectedPurchase); }}><FaEdit /> Edit</button>
                            <button className="btn btn-secondary" onClick={() => window.print()}><FaPrint /> Print</button>
                        </div>
                    </div>
                </div>
            )}
            <style>{`
                @media print {
                    body * { visibility: hidden; }
                    .modal-overlay, .modal-content, #purchase-print-area, #purchase-print-area * { visibility: visible; }
                    .modal-overlay { position: absolute; left: 0; top: 0; background: white; }
                    .modal-content { box-shadow: none; border: none; width: 100%; }
                    .no-print { display: none; }
                }
            `}</style>
        </div>
    );
};

export default Purchases;
