import React, { useState, useRef } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';
import { FaPlus, FaTrash, FaPrint, FaSearch } from 'react-icons/fa';
import { useReactToPrint } from 'react-to-print';
import Barcode from 'react-barcode';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';

const ReturnInvoice = React.forwardRef(({ returnData, customers, suppliers }, ref) => {
    const { formatPrice } = useSettings();
    if (!returnData) return null;
    const isSale = returnData.type === 'sale';
    const entityName = isSale
        ? customers?.find(c => c.id === returnData.customer_id)?.name
        : suppliers?.find(s => s.id === returnData.supplier_id)?.name;

    return (
        <div ref={ref} style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h2 style={{ textAlign: 'center' }}>{isSale ? 'Sales Return' : 'Purchase Return'} Note</h2>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                <p><strong>Reference:</strong> {returnData.reference_no}</p>
                <p><strong>Date:</strong> {new Date(returnData.date).toLocaleString()}</p>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <p><strong>{isSale ? 'Customer' : 'Supplier'}:</strong> {entityName || 'Walk-in'}</p>
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                <thead>
                    <tr style={{ borderBottom: '2px solid #000' }}>
                        <th style={{ textAlign: 'left', padding: '5px' }}>Item</th>
                        <th style={{ textAlign: 'right', padding: '5px' }}>Quantity</th>
                        <th style={{ textAlign: 'right', padding: '5px' }}>Refund Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {returnData.items?.map((item, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                            <td style={{ padding: '5px' }}>{item.name} ({item.code})</td>
                            <td style={{ padding: '5px', textAlign: 'right' }}>{item.quantity}</td>
                            <td style={{ padding: '5px', textAlign: 'right' }}>{formatPrice(item.price * item.quantity)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <div style={{ marginTop: '30px', textAlign: 'right' }}>
                <h4>Total Refund: {formatPrice(returnData.totalRefund)}</h4>
            </div>

            {returnData.staff_note && (
                <div style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '10px' }}>
                    <p><strong>Note:</strong> {returnData.staff_note}</p>
                </div>
            )}

            <div style={{ marginTop: '30px', textAlign: 'center' }}>
                <Barcode value={returnData.reference_no} height={40} />
            </div>
        </div>
    );
});

const Returns = () => {
    const returns = useLiveQuery(() => db.returns.orderBy('date').reverse().toArray());
    const customers = useLiveQuery(() => db.customers.toArray());
    const suppliers = useLiveQuery(() => db.suppliers.toArray());
    const products = useLiveQuery(() => db.products.toArray());
    const sales = useLiveQuery(() => db.sales.orderBy('date').reverse().toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());
    const { formatPrice } = useSettings();
    const { hasPermission } = useAuth();

    const [showForm, setShowForm] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');

    // Form State
    const [formData, setFormData] = useState({
        type: 'sale', // or 'purchase'
        transaction_id: '', // optional reference to original transaction
        warehouse_id: '',
        customer_id: '',
        supplier_id: '',
        reference_no: '',
        items: [],
        staff_note: ''
    });

    const [productSearch, setProductSearch] = useState('');

    // Print Logic
    const componentRef = useRef();
    const [printReturn, setPrintReturn] = useState(null);
    const handlePrint = useReactToPrint({
        contentRef: componentRef,
        onAfterPrint: () => setPrintReturn(null)
    });

    const initiatePrint = (ret) => {
        setPrintReturn(ret);
        setTimeout(handlePrint, 100);
    };

    const handleTransactionChange = async (txId, type) => {
        // Auto-populate items from transaction
        if (!txId) return;

        let tx;
        if (type === 'sale') {
            tx = sales.find(s => s.id === parseInt(txId));
            if (tx) {
                setFormData(prev => ({
                    ...prev,
                    transaction_id: txId,
                    customer_id: tx.customer_id,
                    warehouse_id: tx.warehouse_id || ''
                }));
            }
        } else {
            tx = purchases.find(p => p.id === parseInt(txId));
            if (tx) {
                setFormData(prev => ({
                    ...prev,
                    transaction_id: txId,
                    supplier_id: tx.supplier_id,
                    warehouse_id: tx.warehouse_id || ''
                }));
            }
        }
    };

    const addItem = (product) => {
        const existing = formData.items.find(i => i.id === product.id);
        if (existing) {
            setFormData({
                ...formData,
                items: formData.items.map(i => i.id === product.id ? { ...i, quantity: i.quantity + 1 } : i)
            });
        } else {
            setFormData({
                ...formData,
                items: [...formData.items, { ...product, quantity: 1, price: product.price }]
            }); // Use current product price as default refund price
        }
    };

    const updateItem = (id, field, value) => {
        const numVal = parseFloat(value);
        if (field === 'quantity' && numVal < 1) return;
        if (field === 'price' && numVal < 0) return;
        
        setFormData({
            ...formData,
            items: formData.items.map(i => i.id === id ? { ...i, [field]: isNaN(numVal) ? 0 : numVal } : i)
        });
    };

    const removeItem = (id) => {
        setFormData({ ...formData, items: formData.items.filter(i => i.id !== id) });
    };

    const handleSave = async (e) => {
        e.preventDefault();
        if (!formData.warehouse_id) return alert('Please select a warehouse');
        if (formData.items.length === 0) return alert('Add items to return');
        
        const hasInvalidItem = formData.items.some(i => i.quantity < 1 || i.price < 0);
        if (hasInvalidItem) return alert('All items must have quantity >= 1 and price >= 0');

        const totalRefund = formData.items.reduce((sum, item) => sum + (item.price * item.quantity), 0);

        const returnData = {
            ...formData,
            reference_no: formData.reference_no || `RET-${Date.now()}`,
            date: new Date(),
            transaction_id: parseInt(formData.transaction_id) || null,
            customer_id: parseInt(formData.customer_id) || null,
            supplier_id: parseInt(formData.supplier_id) || null,
            totalRefund,
            status: 'completed'
        };

        const returnId = await db.returns.add(returnData);

        // Adjust Stock
        const multiplier = formData.type === 'sale' ? 1 : -1;
        const whId = parseInt(formData.warehouse_id || 1);

        for (const item of formData.items) {
            // 1. Update Warehouse Stock
            const existingStock = await db.stock
                .where('product_id').equals(item.id)
                .filter(s => s.warehouse_id === whId)
                .first();

            if (existingStock) {
                await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) + (item.quantity * multiplier) });
            } else {
                await db.stock.add({
                    product_id: item.id,
                    variant_id: 0,
                    warehouse_id: whId,
                    quantity: item.quantity * multiplier
                });
            }

            // 2. Global Stock Re-sync
            const product = await db.products.get(item.id);
            if (product) {
                const allWhStocks = await db.stock.where('product_id').equals(item.id).toArray();
                const totalWhStock = allWhStocks.reduce((sum, s) => sum + s.quantity, 0);
                await db.products.update(item.id, { stock_quantity: totalWhStock });
            }
        }

        alert(`Return processed. Refund Amount: ${formatPrice(totalRefund)}`);
        setShowForm(false);
        setFormData({ type: 'sale', transaction_id: '', warehouse_id: '', customer_id: '', supplier_id: '', reference_no: '', items: [], staff_note: '' });
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete return record? STOCK WILL NOT BE REVERTED.')) {
            await db.returns.delete(id);
        }
    };

    const filteredReturns = returns?.filter(r =>
        searchTerm === '' || r.reference_no.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Returns</h2>
                {hasPermission('returns_add') && (
                    <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
                        <FaPlus /> Add Return
                    </button>
                )}
            </div>

            <div style={{ display: 'none' }}>
                <ReturnInvoice ref={componentRef} returnData={printReturn} customers={customers} suppliers={suppliers} />
            </div>

            {showForm && (
                <div className="card mb-3">
                    <h3>Create Return</h3>
                    <form onSubmit={handleSave}>
                        <div style={{ display: 'flex', gap: '20px', marginBottom: '15px' }}>
                            <div style={{ flex: 1 }}>
                                <label>Type</label>
                                <select className="form-control" value={formData.type} onChange={e => setFormData({ ...formData, type: e.target.value, transaction_id: '', customer_id: '', supplier_id: '' })}>
                                    <option value="sale">Sale Return</option>
                                    <option value="purchase">Purchase Return</option>
                                </select>
                            </div>
                            <div style={{ flex: 1 }}>
                                <label>{formData.type === 'sale' ? 'Reference Sale (Optional)' : 'Reference Purchase (Optional)'}</label>
                                <select className="form-control" value={formData.transaction_id} onChange={e => handleTransactionChange(e.target.value, formData.type)}>
                                    <option value="">Select Transaction...</option>
                                    {formData.type === 'sale'
                                        ? sales?.map(s => <option key={s.id} value={s.id}>{s.reference_no || `Sale #${s.id}`} ({new Date(s.date).toLocaleDateString()})</option>)
                                        : purchases?.map(p => <option key={p.id} value={p.id}>{p.reference_no || `Purchase #${p.id}`} ({new Date(p.date).toLocaleDateString()})</option>)
                                    }
                                </select>
                            </div>
                            <div style={{ flex: 1 }}>
                                <label>Warehouse</label>
                                <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })} required>
                                    <option value="">Select Warehouse...</option>
                                    {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                                </select>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '20px', marginBottom: '15px' }}>
                            <div style={{ flex: 1 }}>
                                <label>{formData.type === 'sale' ? 'Customer' : 'Supplier'}</label>
                                {formData.type === 'sale' ? (
                                    <select className="form-control" value={formData.customer_id} onChange={e => setFormData({ ...formData, customer_id: e.target.value })} >
                                        <option value="">Walk-in Customer</option>
                                        {customers?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                                    </select>
                                ) : (
                                    <select className="form-control" value={formData.supplier_id} onChange={e => setFormData({ ...formData, supplier_id: e.target.value })} required>
                                        <option value="">Select Supplier</option>
                                        {suppliers?.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                                    </select>
                                )}
                            </div>
                            <div style={{ flex: 1 }}>
                                <label>Reference No</label>
                                <input className="form-control" value={formData.reference_no} onChange={e => setFormData({ ...formData, reference_no: e.target.value })} placeholder="Auto-generated" />
                            </div>
                        </div>

                        <div style={{ marginBottom: '15px', padding: '10px', background: '#f9f9f9', border: '1px solid #eee' }}>
                            <h4>Return Items</h4>
                            <input
                                className="form-control"
                                placeholder="Search products to return..."
                                value={productSearch}
                                onChange={e => setProductSearch(e.target.value)}
                            />
                            {productSearch && (
                                <div style={{ maxHeight: '150px', overflowY: 'auto', background: 'white', border: '1px solid #ddd', marginTop: '5px' }}>
                                    {products?.filter(p => p.name.toLowerCase().includes(productSearch.toLowerCase())).map(p => (
                                        <div
                                            key={p.id}
                                            style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid #eee' }}
                                            onClick={() => { addItem(p); setProductSearch(''); }}
                                        >
                                            {p.name} ({p.code})
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <table style={{ width: '100%', marginBottom: '15px' }}>
                            <thead>
                                <tr style={{ background: '#eee' }}>
                                    <th style={{ padding: '5px', textAlign: 'left' }}>Product</th>
                                    <th style={{ padding: '5px' }}>Price</th>
                                    <th style={{ padding: '5px' }}>Qty (Return)</th>
                                    <th style={{ padding: '5px' }}>Total</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {formData.items.map(item => (
                                    <tr key={item.id} style={{ borderBottom: '1px solid #eee' }}>
                                        <td style={{ padding: '5px' }}>{item.name}</td>
                                        <td style={{ padding: '5px' }}>
                                            <input type="number" className="form-control" style={{ width: '80px' }} value={item.price} onChange={e => updateItem(item.id, 'price', e.target.value)} />
                                        </td>
                                        <td style={{ padding: '5px' }}>
                                            <input type="number" className="form-control" style={{ width: '80px' }} value={item.quantity} onChange={e => updateItem(item.id, 'quantity', e.target.value)} />
                                        </td>
                                        <td style={{ padding: '5px' }}>{formatPrice(item.price * item.quantity)}</td>
                                        <td style={{ padding: '5px', textAlign: 'center' }}>
                                            <FaTrash style={{ color: 'red', cursor: 'pointer' }} onClick={() => removeItem(item.id)} />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <div style={{ marginBottom: '15px' }}>
                            <label>Return Note</label>
                            <textarea className="form-control" value={formData.staff_note} onChange={e => setFormData({ ...formData, staff_note: e.target.value })} rows="2"></textarea>
                        </div>

                        <button type="submit" className="btn btn-primary">Process Refund</button>
                        <button type="button" className="btn btn-secondary" style={{ marginLeft: '10px' }} onClick={() => setShowForm(false)}>Cancel</button>
                    </form>
                </div>
            )}

            <div className="card">
                <div style={{ marginBottom: '15px' }}>
                    <input className="form-control" placeholder="Search return reference..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Reference</th>
                            <th style={{ padding: '10px' }}>Type</th>
                            <th style={{ padding: '10px' }}>Entity</th>
                            <th style={{ padding: '10px' }}>Refund</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredReturns?.map(r => (
                            <tr key={r.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(r.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{r.reference_no}</td>
                                <td style={{ padding: '10px' }}>{r.type === 'purchase' ? 'Purchase Return' : 'Sale Return'}</td>
                                <td style={{ padding: '10px' }}>
                                    {r.type === 'purchase'
                                        ? suppliers?.find(s => s.id === r.supplier_id)?.name
                                        : (customers?.find(c => c.id === r.customer_id)?.name || 'Walk-in')}
                                </td>
                                <td style={{ padding: '10px' }}>{formatPrice(r.totalRefund)}</td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" onClick={() => initiatePrint(r)} title="Print"><FaPrint /></button>
                                    {hasPermission('returns_delete') && (
                                        <button className="btn-icon" style={{ color: 'red', marginLeft: '5px' }} onClick={() => handleDelete(r.id)}><FaTrash /></button>
                                    )}
                                </td>
                            </tr>
                        ))}
                        {filteredReturns?.length === 0 && <tr><td colSpan="6" style={{ padding: '20px', textAlign: 'center' }}>No returns found.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Returns;
