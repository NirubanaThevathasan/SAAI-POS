import React, { useState } from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useAuth } from '../context/AuthContext';
import { FaPlus, FaCheck, FaTimes, FaEdit, FaTrash, FaThumbsUp } from 'react-icons/fa';

const ProductRequests = () => {
    const { user } = useAuth();
    const [showModal, setShowModal] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [formData, setFormData] = useState({
        item_name: '',
        category: '',
        requested_quantity: '',
        customer_name: '',
        phone: '',
        notes: ''
    });

    const requests = useLiveQuery(() => db.product_requests.orderBy('date').reverse().toArray());

    const resetForm = () => {
        setFormData({
            item_name: '',
            category: '',
            requested_quantity: '',
            customer_name: '',
            phone: '',
            notes: ''
        });
        setEditingId(null);
    };

    const handleEdit = (req) => {
        setFormData({
            item_name: req.item_name,
            category: req.category || '',
            requested_quantity: req.requested_quantity || '',
            customer_name: req.customer_name || '',
            phone: req.phone || '',
            notes: req.notes || ''
        });
        setEditingId(req.id);
        setShowModal(true);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const data = {
            ...formData,
            requested_quantity: parseFloat(formData.requested_quantity) || 0
        };

        if (editingId) {
            await db.product_requests.update(editingId, data);
        } else {
            await db.product_requests.add({
                ...data,
                date: new Date(),
                status: 'pending',
                staff_name: user?.username || 'Unknown'
            });
        }

        resetForm();
        setShowModal(false);
    };

    const updateStatus = async (id, status) => {
        await db.product_requests.update(id, { status });
    };

    const deleteRequest = async (id) => {
        if (window.confirm('Delete this request?')) {
            await db.product_requests.delete(id);
        }
    };

    const handleCloseModal = () => {
        setShowModal(false);
        resetForm();
    };

    return (
        <div className="page-content">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                    <h2 style={{ margin: 0 }}>Customer Suggestions / Product Requests</h2>
                    <p style={{ margin: '5px 0 0', color: '#666' }}>Track items customers are asking for that are out of stock or new.</p>
                </div>
                <button className="btn btn-primary" onClick={() => { resetForm(); setShowModal(true); }}>
                    <FaPlus /> New Suggestion
                </button>
            </div>

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                            <th style={{ padding: '12px' }}>Date</th>
                            <th style={{ padding: '12px' }}>Item Name</th>
                            <th style={{ padding: '12px' }}>Customer Info</th>
                            <th style={{ padding: '12px' }}>Notes</th>
                            <th style={{ padding: '12px' }}>Staff</th>
                            <th style={{ padding: '12px' }}>Status</th>
                            <th style={{ padding: '12px', textAlign: 'right' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {requests?.map(req => (
                            <tr key={req.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                                <td style={{ padding: '12px', fontSize: '0.9rem' }}>{new Date(req.date).toLocaleDateString()}</td>
                                <td style={{ padding: '12px', fontWeight: 'bold' }}>
                                    {req.item_name}
                                    {req.category && <span style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'normal', color: '#666' }}>{req.category}</span>}
                                </td>
                                <td style={{ padding: '12px' }}>
                                    <div>{req.customer_name || 'N/A'}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#666' }}>{req.phone}</div>
                                </td>
                                <td style={{ padding: '12px', maxWidth: '200px' }}>
                                    {req.notes}
                                    {req.requested_quantity > 0 && <div style={{ fontSize: '0.8rem', color: '#6366f1' }}>Req Qty: {req.requested_quantity}</div>}
                                </td>
                                <td style={{ padding: '12px', fontSize: '0.9rem' }}>{req.staff_name}</td>
                                <td style={{ padding: '12px' }}>
                                    <span style={{
                                        padding: '4px 10px', borderRadius: '12px', fontSize: '0.8rem', textTransform: 'uppercase', fontWeight: '600',
                                        background: req.status === 'pending' ? '#fef3c7' : req.status === 'added' ? '#d1fae5' : '#fee2e2',
                                        color: req.status === 'pending' ? '#d97706' : req.status === 'added' ? '#059669' : '#dc2626'
                                    }}>
                                        {req.status}
                                    </span>
                                </td>
                                <td style={{ padding: '12px', textAlign: 'right' }}>
                                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                                        {req.status === 'pending' && (
                                            <>
                                                <button onClick={() => updateStatus(req.id, 'added')} className="btn" style={{ padding: '6px', background: '#10b981', color: 'white', borderRadius: '4px' }} title="Mark Added"><FaCheck /></button>
                                                <button onClick={() => updateStatus(req.id, 'ignored')} className="btn" style={{ padding: '6px', background: '#f59e0b', color: 'white', borderRadius: '4px' }} title="Ignore"><FaTimes /></button>
                                            </>
                                        )}
                                        <button onClick={() => handleEdit(req)} className="btn" style={{ padding: '6px', background: '#6366f1', color: 'white', borderRadius: '4px' }} title="Edit"><FaEdit /></button>
                                        <button onClick={() => deleteRequest(req.id)} className="btn" style={{ padding: '6px', background: '#ef4444', color: 'white', borderRadius: '4px' }} title="Delete"><FaTrash /></button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {requests?.length === 0 && <tr><td colSpan="7" style={{ padding: '20px', textAlign: 'center', color: '#666' }}>No suggestions recorded yet.</td></tr>}
                    </tbody>
                </table>
            </div>

            {showModal && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
                }}>
                    <div className="modal-content card" style={{ width: '500px', padding: '24px' }}>
                        <h3 style={{ marginTop: 0 }}>{editingId ? 'Edit Suggestion' : 'Record New Suggestion'}</h3>
                        <form onSubmit={handleSubmit}>
                            <div style={{ marginBottom: '15px' }}>
                                <label className="form-label">Item Name *</label>
                                <input required type="text" className="form-control" value={formData.item_name} onChange={e => setFormData({ ...formData, item_name: e.target.value })} placeholder="e.g. Wireless Mouse Model X" />
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                                <div>
                                    <label className="form-label">Category</label>
                                    <input type="text" className="form-control" value={formData.category} onChange={e => setFormData({ ...formData, category: e.target.value })} />
                                </div>
                                <div>
                                    <label className="form-label">Req. Qty *</label>
                                    <input type="number" min="1" step="1" required className="form-control" value={formData.requested_quantity} onChange={e => setFormData({ ...formData, requested_quantity: e.target.value })} placeholder="1" />
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                                <div>
                                    <label className="form-label">Customer Name</label>
                                    <input type="text" className="form-control" value={formData.customer_name} onChange={e => setFormData({ ...formData, customer_name: e.target.value })} placeholder="Customer Name" />
                                </div>
                                <div>
                                    <label className="form-label">Phone</label>
                                    <input type="tel" pattern="^(?:0|94|\+94|0094)?[1-9][0-9]{8}$" title="Valid Sri Lankan phone number" className="form-control" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} placeholder="0771234567" />
                                </div>
                            </div>
                            <div style={{ marginBottom: '20px' }}>
                                <label className="form-label">Notes</label>
                                <textarea className="form-control" rows="3" value={formData.notes} onChange={e => setFormData({ ...formData, notes: e.target.value })} placeholder="Any specific requirements..."></textarea>
                            </div>
                            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                                <button type="button" className="btn btn-secondary" onClick={handleCloseModal}>Cancel</button>
                                <button type="submit" className="btn btn-primary">{editingId ? 'Update' : 'Save Suggestion'}</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ProductRequests;
