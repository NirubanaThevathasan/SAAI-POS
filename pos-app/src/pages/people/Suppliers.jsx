import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaTrash, FaPlus, FaEdit, FaHistory } from 'react-icons/fa';
import { useAuth } from '../../context/AuthContext';

const SupplierHistory = ({ supplierId, onClose }) => {
    const purchases = useLiveQuery(
        () => db.purchases.where('supplier_id').equals(supplierId).reverse().sortBy('date'),
        [supplierId]
    );

    const totalSpent = purchases?.reduce((sum, s) => sum + (s.grandTotal || 0), 0) || 0;
    const pendingPurchases = purchases?.filter(s => s.payment_status !== 'paid').length || 0;

    if (!supplierId) return null;

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
                            Total Orders: <strong>{purchases?.length || 0}</strong> |
                            Total Value: <strong>${totalSpent.toFixed(2)}</strong> |
                            Pending/Partial: <strong>{pendingPurchases}</strong>
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
                        {purchases?.map(purchase => (
                            <tr key={purchase.id} style={{ borderBottom: '1px solid #eee' }}>
                                <td style={{ padding: '8px' }}>{new Date(purchase.date).toLocaleDateString()}</td>
                                <td style={{ padding: '8px' }}>{purchase.reference_no}</td>
                                <td style={{ padding: '8px' }}>{purchase.status}</td>
                                <td style={{ padding: '8px' }}>{purchase.payment_status}</td>
                                <td style={{ padding: '8px' }}>${purchase.grandTotal?.toFixed(2) || '0.00'}</td>
                            </tr>
                        ))}
                        {purchases?.length === 0 && <tr><td colSpan="5" style={{ padding: '20px', textAlign: 'center' }}>No purchases found.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

const Suppliers = () => {
    const { hasPermission } = useAuth();
    const suppliers = useLiveQuery(() => db.suppliers.toArray());
    const [showForm, setShowForm] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [viewHistoryId, setViewHistoryId] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');

    const filteredSuppliers = suppliers?.filter(s =>
        s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.phone.includes(searchTerm) ||
        s.email.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phone: '',
        city: '',
        address: '',
        tax_number: '',
        payment_terms: ''
    });

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            if (editingId) {
                await db.suppliers.update(editingId, formData);
                setEditingId(null);
            } else {
                await db.suppliers.add(formData);
            }
            resetForm();
        } catch (error) {
            alert('Error saving supplier');
        }
    };

    const handleEdit = (supplier) => {
        setFormData({
            name: supplier.name,
            email: supplier.email,
            phone: supplier.phone,
            city: supplier.city,
            address: supplier.address || '',
            tax_number: supplier.tax_number || '',
            payment_terms: supplier.payment_terms || ''
        });
        setEditingId(supplier.id);
        setShowForm(true);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete supplier?')) {
            await db.suppliers.delete(id);
        }
    };

    const resetForm = () => {
        setFormData({ name: '', email: '', phone: '', city: '', address: '', tax_number: '', payment_terms: '' });
        setEditingId(null);
        setShowForm(false);
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Suppliers</h2>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        type="text"
                        placeholder="Search suppliers..."
                        className="form-control"
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        style={{ width: '250px' }}
                    />
                    {hasPermission('suppliers_add') && (
                        <button className="btn btn-primary" onClick={() => { resetForm(); setShowForm(!showForm); }}>
                            <FaPlus /> Add Supplier
                        </button>
                    )}
                </div>
            </div>

            {viewHistoryId && <SupplierHistory supplierId={viewHistoryId} onClose={() => setViewHistoryId(null)} />}

            {showForm && (
                <div className="card mb-3">
                    <h3>{editingId ? 'Edit Supplier' : 'Add New Supplier'}</h3>
                    <form onSubmit={handleSubmit}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                            <div className="form-group">
                                <label>Name *</label>
                                <input type="text" className="form-control" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} required placeholder="Company or Individual Name" />
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
                                <input type="text" className="form-control" value={formData.city} onChange={e => setFormData({ ...formData, city: e.target.value })} />
                            </div>
                            <div className="form-group">
                                <label>Address</label>
                                <input type="text" className="form-control" value={formData.address} onChange={e => setFormData({ ...formData, address: e.target.value })} />
                            </div>
                            <div className="form-group">
                                <label>Payment Terms</label>
                                <select className="form-control" value={formData.payment_terms} onChange={e => setFormData({ ...formData, payment_terms: e.target.value })}>
                                    <option value="">Select Terms</option>
                                    <option value="net0">Immediate (Cash/COD)</option>
                                    <option value="net15">Net 15</option>
                                    <option value="net30">Net 30</option>
                                    <option value="net45">Net 45</option>
                                    <option value="net60">Net 60</option>
                                </select>
                            </div>
                        </div>
                        <div style={{ marginTop: '15px' }}>
                            <button type="submit" className="btn btn-primary">{editingId ? 'Update' : 'Save Supplier'}</button>
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
                            <th style={{ padding: '10px' }}>Terms</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredSuppliers?.map(s => (
                            <tr key={s.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>
                                    {s.name}<br />
                                    <small style={{ color: '#888' }}>{s.tax_number}</small>
                                </td>
                                <td style={{ padding: '10px' }}>
                                    {s.phone}<br />
                                    <small style={{ color: '#888' }}>{s.email}</small>
                                </td>
                                <td style={{ padding: '10px' }}>{s.city}</td>
                                <td style={{ padding: '10px' }}>
                                    {s.payment_terms === 'net0' ? 'Immediate' :
                                        s.payment_terms ? s.payment_terms.replace('net', 'Net ') : '-'}
                                </td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" onClick={() => setViewHistoryId(s.id)} title="History"><FaHistory /></button>
                                    {hasPermission('suppliers_edit') && (
                                        <button className="btn-icon" style={{ marginLeft: '5px' }} onClick={() => handleEdit(s)}><FaEdit /></button>
                                    )}
                                    {hasPermission('suppliers_delete') && (
                                        <button className="btn-icon" style={{ color: 'red', marginLeft: '5px' }} onClick={() => handleDelete(s.id)}><FaTrash /></button>
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

export default Suppliers;
