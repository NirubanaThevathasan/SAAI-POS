import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaTrash, FaPlus, FaEdit } from 'react-icons/fa';

/**
 * Generic component for managing people (Customers, Suppliers, etc.)
 * Fields expected: name, email, phone, city
 */
const PersonManager = ({ title, table }) => {
    const people = useLiveQuery(() => db[table].toArray());
    const [showForm, setShowForm] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phone: '',
        city: ''
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (editingId) {
                await db[table].update(editingId, formData);
                setEditingId(null);
            } else {
                await db[table].add(formData);
            }
            setFormData({ name: '', email: '', phone: '', city: '' });
            setShowForm(false);
        } catch (error) {
            console.error(`Failed to save to ${table}:`, error);
            alert('Error saving record.');
        }
    };

    const handleEdit = (person) => {
        setFormData({ name: person.name, email: person.email, phone: person.phone, city: person.city });
        setEditingId(person.id);
        setShowForm(true);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Are you sure you want to delete this record?')) {
            await db[table].delete(id);
        }
    };

    const cancelForm = () => {
        setFormData({ name: '', email: '', phone: '', city: '' });
        setEditingId(null);
        setShowForm(false);
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>{title}</h2>
                <button className="btn btn-primary" onClick={() => { cancelForm(); setShowForm(!showForm); }}>
                    <FaPlus /> Add New
                </button>
            </div>

            {showForm && (
                <div className="card mb-3">
                    <h3>{editingId ? 'Edit' : 'Add New'} {title}</h3>
                    <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                        <div className="form-group">
                            <label>Name *</label>
                            <input
                                type="text"
                                className="form-control"
                                value={formData.name}
                                onChange={e => setFormData({ ...formData, name: e.target.value })}
                                required
                                placeholder="Full Name"
                                style={{ width: '100%', padding: '8px', marginTop: '5px' }}
                            />
                        </div>
                        <div className="form-group">
                            <label>Phone *</label>
                            <input
                                type="tel"
                                className="form-control"
                                value={formData.phone}
                                onChange={e => setFormData({ ...formData, phone: e.target.value })}
                                required
                                pattern="^(?:0|94|\+94|0094)?[1-9][0-9]{8}$"
                                title="Enter a valid Sri Lankan phone number (e.g., 0771234567 or +94771234567)"
                                placeholder="e.g., 0771234567"
                                style={{ width: '100%', padding: '8px', marginTop: '5px' }}
                            />
                        </div>
                        <div className="form-group">
                            <label>Email</label>
                            <input
                                type="email"
                                className="form-control"
                                value={formData.email}
                                onChange={e => setFormData({ ...formData, email: e.target.value })}
                                placeholder="email@example.com"
                                style={{ width: '100%', padding: '8px', marginTop: '5px' }}
                            />
                        </div>
                        <div className="form-group">
                            <label>City</label>
                            <input
                                type="text"
                                className="form-control"
                                value={formData.city}
                                onChange={e => setFormData({ ...formData, city: e.target.value })}
                                style={{ width: '100%', padding: '8px', marginTop: '5px' }}
                            />
                        </div>
                        <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                            <button type="submit" className="btn btn-primary">{editingId ? 'Update' : 'Save'}</button>
                            <button type="button" className="btn" onClick={cancelForm} style={{ marginLeft: '10px' }}>Cancel</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left' }}>
                            <th style={{ padding: '10px' }}>Name</th>
                            <th style={{ padding: '10px' }}>Phone</th>
                            <th style={{ padding: '10px' }}>Email</th>
                            <th style={{ padding: '10px' }}>City</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {people?.map(person => (
                            <tr key={person.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{person.name}</td>
                                <td style={{ padding: '10px' }}>{person.phone}</td>
                                <td style={{ padding: '10px' }}>{person.email}</td>
                                <td style={{ padding: '10px' }}>{person.city}</td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" style={{ cursor: 'pointer', marginRight: '10px' }} onClick={() => handleEdit(person)}>
                                        <FaEdit />
                                    </button>
                                    <button className="btn-icon" style={{ cursor: 'pointer', color: 'red' }} onClick={() => handleDelete(person.id)}>
                                        <FaTrash />
                                    </button>
                                </td>
                            </tr>
                        ))}
                        {people?.length === 0 && (
                            <tr>
                                <td colSpan="5" style={{ padding: '20px', textAlign: 'center' }}>No records found.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PersonManager;
