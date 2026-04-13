import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaTrash, FaPlus, FaEdit, FaEye, FaHistory } from 'react-icons/fa';
import { useAuth } from '../../context/AuthContext';

const CustomerHistory = ({ customerId, onClose }) => {
    const sales = useLiveQuery(
        () => db.sales.where('customer_id').equals(customerId).reverse().sortBy('date'),
        [customerId]
    );

    const totalSpent = sales?.reduce((sum, s) => sum + (s.grandTotal || 0), 0) || 0;
    const pendingSales = sales?.filter(s => s.payment_status !== 'paid').length || 0;

    if (!customerId) return null;

    return (
        <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
        }}>
            <div style={{
                background: 'white', padding: '20px', borderRadius: '8px', width: '80%', maxWidth: '800px',
                maxHeight: '90vh', overflowY: 'auto'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
                    <div>
                        <h3>Purchase History</h3>
                        <p style={{ margin: 0, color: '#666' }}>
                            Total Orders: <strong>{sales?.length || 0}</strong> |
                            Total Value: <strong>${totalSpent.toFixed(2)}</strong> |
                            Pending/Partial: <strong>{pendingSales}</strong>
                        </p>
                    </div>
                    <button className="btn btn-secondary" onClick={onClose} style={{ height: 'fit-content' }}>Close</button>
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ background: '#eee', textAlign: 'left' }}>
                            <th style={{ padding: '8px' }}>Date</th>
                            <th style={{ padding: '8px' }}>Ref</th>
                            <th style={{ padding: '8px' }}>Status</th>
                            <th style={{ padding: '8px' }}>Payment</th>
                            <th style={{ padding: '8px' }}>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sales?.map(sale => (
                            <tr key={sale.id} style={{ borderBottom: '1px solid #eee' }}>
                                <td style={{ padding: '8px' }}>{new Date(sale.date).toLocaleDateString()}</td>
                                <td style={{ padding: '8px' }}>{sale.reference_no}</td>
                                <td style={{ padding: '8px' }}>{sale.status}</td>
                                <td style={{ padding: '8px' }}>{sale.payment_status}</td>
                                <td style={{ padding: '8px' }}>${sale.grandTotal?.toFixed(2) || '0.00'}</td>
                            </tr>
                        ))}
                        {sales?.length === 0 && <tr><td colSpan="5" style={{ padding: '20px', textAlign: 'center' }}>No purchases found.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

const Customers = () => {
    const { hasPermission } = useAuth();
    const customers = useLiveQuery(() => db.customers.toArray());
    const [showForm, setShowForm] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [viewHistoryId, setViewHistoryId] = useState(null);

    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phone: '',
        city: '',
        address: '',
        tax_number: '',
        credit_limit: '',
        discount: ''
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        const dataToSave = {
            ...formData,
            credit_limit: parseFloat(formData.credit_limit) || 0,
            discount: parseFloat(formData.discount) || 0
        };

        try {
            if (editingId) {
                await db.customers.update(editingId, dataToSave);
                setEditingId(null);
            } else {
                await db.customers.add(dataToSave);
            }
            resetForm();
        } catch (error) {
            alert('Error saving customer');
        }
    };

    const handleEdit = (customer) => {
        setFormData({
            name: customer.name,
            email: customer.email,
            phone: customer.phone,
            city: customer.city,
            address: customer.address || '',
            tax_number: customer.tax_number || '',
            credit_limit: customer.credit_limit || '',
            discount: customer.discount || ''
        });
        setEditingId(customer.id);
        setShowForm(true);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete customer?')) {
            await db.customers.delete(id);
        }
    };

    const resetForm = () => {
        setFormData({ name: '', email: '', phone: '', city: '', address: '', tax_number: '', credit_limit: '', discount: '' });
        setEditingId(null);
        setShowForm(false);
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Customers</h2>
                {hasPermission('customers_add') && (
                    <button className="btn btn-primary" onClick={() => { resetForm(); setShowForm(!showForm); }}>
                        <FaPlus /> Add Customer
                    </button>
                )}
            </div>

            {viewHistoryId && <CustomerHistory customerId={viewHistoryId} onClose={() => setViewHistoryId(null)} />}

            {showForm && (
                <div className="card mb-3">
                    <h3>{editingId ? 'Edit Customer' : 'Add New Customer'}</h3>
                    <form onSubmit={handleSubmit}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                            <div className="form-group">
                                <label>Name *</label>
                                <input type="text" className="form-control" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} required placeholder="Full Name" />
                            </div>
                            <div className="form-group">
                                <label>Email</label>
                                <input type="email" className="form-control" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} placeholder="email@example.com" />
                            </div>
                            <div className="form-group">
                                <label>Phone *</label>
                                <input type="tel" className="form-control" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} required pattern="^(?:0|94|\+94|0094)?[1-9][0-9]{8}$" title="Valid Sri Lankan phone number (e.g., 0771234567)" placeholder="e.g., 0771234567" />
                            </div>
                            <div className="form-group">
                                <label>Tax Number</label>
                                <input type="text" className="form-control" value={formData.tax_number} onChange={e => setFormData({ ...formData, tax_number: e.target.value })} placeholder="Tax ID" />
                            </div>
                            <div className="form-group">
                                <label>City</label>
                                <input type="text" className="form-control" value={formData.city} onChange={e => setFormData({ ...formData, city: e.target.value })} placeholder="City" />
                            </div>
                            <div className="form-group">
                                <label>Address</label>
                                <input type="text" className="form-control" value={formData.address} onChange={e => setFormData({ ...formData, address: e.target.value })} placeholder="Full Address" />
                            </div>
                            <div className="form-group">
                                <label>Credit Limit</label>
                                <input type="number" step="0.01" min="0" className="form-control" value={formData.credit_limit} onChange={e => setFormData({ ...formData, credit_limit: e.target.value })} placeholder="0.00" />
                            </div>
                            <div className="form-group">
                                <label>Default Discount (%)</label>
                                <input type="number" step="0.01" min="0" max="100" className="form-control" value={formData.discount} onChange={e => setFormData({ ...formData, discount: e.target.value })} placeholder="0" />
                            </div>
                        </div>
                        <div style={{ marginTop: '15px' }}>
                            <button type="submit" className="btn btn-primary">{editingId ? 'Update' : 'Save Customer'}</button>
                            <button type="button" className="btn" onClick={resetForm} style={{ marginLeft: '10px' }}>Cancel</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Name</th>
                            <th style={{ padding: '10px' }}>Contact</th>
                            <th style={{ padding: '10px' }}>City</th>
                            <th style={{ padding: '10px' }}>Credit Limit</th>
                            <th style={{ padding: '10px' }}>Discount</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {customers?.map(c => (
                            <tr key={c.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>
                                    {c.name}<br />
                                    <small style={{ color: '#888' }}>{c.tax_number}</small>
                                </td>
                                <td style={{ padding: '10px' }}>
                                    {c.phone}<br />
                                    <small style={{ color: '#888' }}>{c.email}</small>
                                </td>
                                <td style={{ padding: '10px' }}>{c.city}</td>
                                <td style={{ padding: '10px' }}>${c.credit_limit?.toFixed(2)}</td>
                                <td style={{ padding: '10px' }}>{c.discount}%</td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" onClick={() => setViewHistoryId(c.id)} title="History"><FaHistory /></button>
                                    {hasPermission('customers_edit') && (
                                        <button className="btn-icon" style={{ marginLeft: '5px' }} onClick={() => handleEdit(c)}><FaEdit /></button>
                                    )}
                                    {hasPermission('customers_delete') && (
                                        <button className="btn-icon" style={{ color: 'red', marginLeft: '5px' }} onClick={() => handleDelete(c.id)}><FaTrash /></button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Customers;
