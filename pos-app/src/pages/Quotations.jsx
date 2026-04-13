import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';
import { FaPlus, FaTrash, FaEye, FaPrint, FaExchangeAlt, FaSearch } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';

const Quotations = () => {
    const { formatPrice, currency } = useSettings();
    const { hasPermission } = useAuth();
    const navigate = useNavigate();
    const quotations = useLiveQuery(() => db.quotations.orderBy('date').reverse().toArray());
    const customers = useLiveQuery(() => db.customers.toArray());
    const products = useLiveQuery(() => db.products.toArray());

    const [view, setView] = useState('list'); // 'list' | 'form'
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedQuote, setSelectedQuote] = useState(null);

    // Form State
    const [formData, setFormData] = useState({
        customer_id: '',
        valid_until: '',
        terms: 'This quotation is valid for 30 days. Goods once sold will not be taken back.',
        status: 'pending'
    });
    const [cart, setCart] = useState([]);

    // --- Actions ---
    const handleAddToCart = (productId) => {
        const product = products.find(p => p.id === parseInt(productId));
        if (!product) return;
        const existing = cart.find(c => c.id === product.id);
        if (existing) {
            setCart(cart.map(c => c.id === product.id ? { ...c, quantity: c.quantity + 1 } : c));
        } else {
            setCart([...cart, { ...product, quantity: 1, price: product.price }]);
        }
    };

    const updateCartItem = (id, field, value) => {
        const numVal = parseFloat(value);
        if (field === 'quantity' && numVal < 1) return;
        if (field === 'price' && numVal < 0) return;
        setCart(cart.map(item => item.id === id ? { ...item, [field]: isNaN(numVal) ? 0 : numVal } : item));
    };

    const removeCartItem = (id) => setCart(cart.filter(item => item.id !== id));

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!formData.customer_id) return alert('Please select a customer');
        if (!formData.valid_until) return alert('Please select a validity date');
        if (cart.length === 0) return alert('Add items to the quotation first');

        const hasInvalidItem = cart.some(i => i.quantity < 1 || i.price < 0);
        if (hasInvalidItem) return alert('All items must have quantity >= 1 and price >= 0');

        const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);

        await db.quotations.add({
            ...formData,
            customer_id: parseInt(formData.customer_id),
            date: new Date(),
            expiry_date: formData.valid_until,
            grand_total: total,
            items: cart
        });

        alert('Quotation Created!');
        setCart([]);
        setFormData({ customer_id: '', valid_until: '', terms: '', status: 'pending' });
        setView('list');
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete this quotation?')) {
            await db.quotations.delete(id);
        }
    };

    const handleConvertToSale = async (quote) => {
        if (!window.confirm('Convert this quotation to a Sale? Stock will be deducted.')) return;

        // Create Sale
        await db.sales.add({
            date: new Date(),
            customer_id: quote.customer_id,
            total: quote.grand_total,
            status: 'completed',
            paymentMethod: 'Cash', // Default to cash for now
            items: quote.items,
            source: 'quotation_converted',
            ref_quotation_id: quote.id
        });

        // Deduct Stock
        for (const item of quote.items) {
            const product = await db.products.get(item.id);
            if (product) {
                await db.products.update(item.id, {
                    stock_quantity: (product.stock_quantity || 0) - item.quantity
                });
            }
        }

        // Update Quote Status
        await db.quotations.update(quote.id, { status: 'converted' });

        alert('Converted to Sale successfully!');
        setSelectedQuote(null);
        navigate('/pos'); // Or stay here
    };

    // --- Render Helpers ---
    const filteredQuotations = quotations?.filter(q =>
        searchTerm === '' || q.id.toString().includes(searchTerm) ||
        (customers?.find(c => c.id === q.customer_id)?.name.toLowerCase().includes(searchTerm.toLowerCase()))
    );

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Quotations</h2>
                {view === 'list' ?
                    (hasPermission('quotations_add') && <button className="btn btn-primary" onClick={() => setView('form')}><FaPlus /> New Quotation</button>) :
                    <button className="btn btn-secondary" onClick={() => setView('list')}>Back to List</button>
                }
            </div>

            {view === 'list' && (
                <div className="card">
                    <div className="mb-3" style={{ maxWidth: '300px' }}>
                        <input className="form-control" placeholder="Search ID or Customer..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                                <th style={{ padding: '10px' }}>Date</th>
                                <th style={{ padding: '10px' }}>Customer</th>
                                <th style={{ padding: '10px' }}>Expiry</th>
                                <th style={{ padding: '10px' }}>Total</th>
                                <th style={{ padding: '10px' }}>Status</th>
                                <th style={{ padding: '10px' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredQuotations?.map(q => (
                                <tr key={q.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                    <td style={{ padding: '10px' }}>{new Date(q.date).toLocaleDateString()}</td>
                                    <td style={{ padding: '10px' }}>{customers?.find(c => c.id === q.customer_id)?.name || 'Guest'}</td>
                                    <td style={{ padding: '10px' }}>{q.expiry_date}</td>
                                    <td style={{ padding: '10px' }}>{formatPrice(q.grand_total)}</td>
                                    <td style={{ padding: '10px' }}>
                                        <span style={{ padding: '4px 8px', borderRadius: '4px', background: q.status === 'converted' ? '#d4edda' : '#fff3cd' }}>
                                            {q.status}
                                        </span>
                                    </td>
                                    <td style={{ padding: '10px', display: 'flex', gap: '5px' }}>
                                        <button className="btn-icon" onClick={() => setSelectedQuote(q)}><FaEye /></button>
                                        {hasPermission('quotations_delete') && (
                                            <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(q.id)}><FaTrash /></button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {view === 'form' && (
                <div className="card">
                    <h3>Create Quotation</h3>
                    <form onSubmit={handleSubmit}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                            <div>
                                <label>Customer</label>
                                <select className="form-control" required value={formData.customer_id} onChange={e => setFormData({ ...formData, customer_id: e.target.value })} style={{ width: '100%' }}>
                                    <option value="">Select Customer</option>
                                    {customers?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                                </select>
                            </div>
                            <div>
                                <label>Valid Until</label>
                                <input type="date" className="form-control" required value={formData.valid_until} onChange={e => setFormData({ ...formData, valid_until: e.target.value })} style={{ width: '100%' }} />
                            </div>
                        </div>

                        <div className="mb-3">
                            <label>Add Product</label>
                            <select className="form-control" onChange={e => { handleAddToCart(e.target.value); e.target.value = ''; }}>
                                <option value="">Select Product...</option>
                                {products?.map(p => <option key={p.id} value={p.id}>{p.name} ({formatPrice(p.price)})</option>)}
                            </select>
                        </div>

                        <table className="table mb-3">
                            <thead><tr><th>Item</th><th>Price</th><th>Qty</th><th>Total</th><th>Action</th></tr></thead>
                            <tbody>
                                {cart.map(item => (
                                    <tr key={item.id}>
                                        <td>{item.name}</td>
                                        <td><input type="number" style={{ width: '80px' }} value={item.price} onChange={e => updateCartItem(item.id, 'price', e.target.value)} /></td>
                                        <td><input type="number" style={{ width: '60px' }} value={item.quantity} onChange={e => updateCartItem(item.id, 'quantity', e.target.value)} /></td>
                                        <td>{formatPrice(item.price * item.quantity)}</td>
                                        <td><button type="button" onClick={() => removeCartItem(item.id)} style={{ color: 'red', border: 'none', background: 'none' }}><FaTrash /></button></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <div className="mb-3">
                            <label>Terms & Conditions</label>
                            <textarea className="form-control" rows="3" value={formData.terms} onChange={e => setFormData({ ...formData, terms: e.target.value })}></textarea>
                        </div>

                        <button type="submit" className="btn btn-primary">Save Quotation</button>
                    </form>
                </div>
            )}

            {/* Quote Details Modal */}
            {selectedQuote && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)',
                    display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
                }}>
                    <div className="modal-content card" style={{ width: '600px', maxHeight: '90vh', overflowY: 'auto', padding: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <h3>Quotation #{selectedQuote.id}</h3>
                            <button className="btn" onClick={() => setSelectedQuote(null)}>Close</button>
                        </div>

                        <div id="quote-print-area" style={{ padding: '20px', border: '1px solid #eee', margin: '20px 0' }}>
                            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                                <h1>QUOTATION</h1>
                                <p>Date: {new Date(selectedQuote.date).toLocaleDateString()}</p>
                                <p>Valid Until: {selectedQuote.expiry_date}</p>
                                <p>Customer: {customers?.find(c => c.id === selectedQuote.customer_id)?.name}</p>
                            </div>
                            <table style={{ width: '100%', marginBottom: '20px' }}>
                                <thead><tr style={{ borderBottom: '1px solid #000' }}><th>Item</th><th>Qty</th><th>Price</th><th>Total</th></tr></thead>
                                <tbody>
                                    {selectedQuote.items.map((item, i) => (
                                        <tr key={i}>
                                            <td>{item.name}</td>
                                            <td>{item.quantity}</td>
                                            <td>{formatPrice(item.price)}</td>
                                            <td>{formatPrice(item.price * item.quantity)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            <div style={{ textAlign: 'right' }}>
                                <h3>Total: {formatPrice(selectedQuote.grand_total)}</h3>
                            </div>
                            <div style={{ marginTop: '20px', fontSize: '12px' }}>
                                <strong>Terms:</strong>
                                <p>{selectedQuote.terms}</p>
                            </div>
                        </div>

                        <div className="no-print" style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                            <button className="btn btn-secondary" onClick={() => window.print()}><FaPrint /> Print</button>
                            {selectedQuote.status !== 'converted' && (
                                <button className="btn btn-success" onClick={() => handleConvertToSale(selectedQuote)}><FaExchangeAlt /> Convert to Sale</button>
                            )}
                        </div>
                    </div>
                </div>
            )}
            <style>{`
                @media print {
                    body * { visibility: hidden; }
                    .modal-overlay, .modal-content, #quote-print-area, #quote-print-area * { visibility: visible; }
                    .modal-overlay { position: absolute; left: 0; top: 0; background: white; }
                    .modal-content { box-shadow: none; border: none; width: 100%; }
                    .no-print { display: none; }
                }
            `}</style>
        </div>
    );
};

export default Quotations;
