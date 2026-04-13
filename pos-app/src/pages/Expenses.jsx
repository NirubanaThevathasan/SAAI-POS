import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';
import { FaPlus, FaTrash, FaSearch } from 'react-icons/fa';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';

const Expenses = () => {
    const { formatPrice, currency } = useSettings();
    const { hasPermission } = useAuth();
    const expenses = useLiveQuery(() => db.expenses.orderBy('date').reverse().toArray());
    const categories = useLiveQuery(() => db.expense_categories.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [showForm, setShowForm] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [dateRange, setDateRange] = useState({ start: '', end: '' });
    const [filterCategory, setFilterCategory] = useState('');

    const [formData, setFormData] = useState({
        reference: '',
        amount: '',
        category_id: '',
        warehouse_id: '',
        note: ''
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        await db.expenses.add({
            ...formData,
            amount: parseFloat(formData.amount),
            category_id: parseInt(formData.category_id),
            warehouse_id: parseInt(formData.warehouse_id) || null,
            date: new Date()
        });
        setFormData({ reference: '', amount: '', category_id: '', warehouse_id: '', note: '' });
        setShowForm(false);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete expense?')) await db.expenses.delete(id);
    };

    const filteredExpenses = expenses?.filter(e => {
        const eDate = new Date(e.date);
        const start = dateRange.start ? new Date(dateRange.start) : null;
        const end = dateRange.end ? new Date(dateRange.end) : null;
        if (end) end.setHours(23, 59, 59);

        const matchesSearch = searchTerm === '' ||
            (e.reference?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                e.note?.toLowerCase().includes(searchTerm.toLowerCase()));

        const matchesCategory = filterCategory ? e.category_id === parseInt(filterCategory) : true;
        const matchesDate = (!start || eDate >= start) && (!end || eDate <= end);

        return matchesSearch && matchesCategory && matchesDate;
    });

    const totalExpense = filteredExpenses?.reduce((sum, e) => sum + e.amount, 0) || 0;

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Expenses</h2>
                {hasPermission('expenses_add') && (
                    <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
                        <FaPlus /> Add Expense
                    </button>
                )}
            </div>

            {showForm && (
                <div className="card mb-3">
                    <h3>Add Expense</h3>
                    <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                        <div>
                            <label>Reference</label>
                            <input type="text" className="form-control" value={formData.reference} onChange={e => setFormData({ ...formData, reference: e.target.value })} style={{ width: '100%', padding: '8px' }} />
                        </div>
                        <div>
                            <label>Amount</label>
                            <div style={{ position: 'relative' }}>
                                <span style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#64748b', fontWeight: 'bold' }}>{currency.symbol}</span>
                                <input type="number" min="0.01" step="0.01" className="form-control" value={formData.amount} onChange={e => setFormData({ ...formData, amount: e.target.value })} required style={{ width: '100%', padding: '8px 8px 8px 35px' }} />
                            </div>
                        </div>
                        <div>
                            <label>Category</label>
                            <select className="form-control" value={formData.category_id} onChange={e => setFormData({ ...formData, category_id: e.target.value })} required style={{ width: '100%', padding: '8px' }}>
                                <option value="">Select Category</option>
                                {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                            </select>
                        </div>
                        <div>
                            <label>Warehouse</label>
                            <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                <option value="">Select Warehouse (Optional)</option>
                                {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                            </select>
                        </div>
                        <div style={{ gridColumn: '1 / -1' }}>
                            <label>Note</label>
                            <input type="text" className="form-control" value={formData.note} onChange={e => setFormData({ ...formData, note: e.target.value })} style={{ width: '100%', padding: '8px' }} />
                        </div>
                        <div style={{ gridColumn: '1 / -1' }}>
                            <button type="submit" className="btn btn-primary">Save Expense</button>
                            <button type="button" className="btn" onClick={() => setShowForm(false)} style={{ marginLeft: '10px' }}>Cancel</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="card">
                {/* Filters */}
                <div className="mb-3 p-3 bg-light rounded" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    <div>
                        <label>Search</label>
                        <input className="form-control" placeholder="Reference or Note..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                    </div>
                    <div>
                        <label>Category</label>
                        <select className="form-control" value={filterCategory} onChange={e => setFilterCategory(e.target.value)}>
                            <option value="">All Categories</option>
                            {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
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
                    <div style={{ marginLeft: 'auto' }}>
                        <h4>Total: {formatPrice(totalExpense)}</h4>
                    </div>
                </div>

                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Reference</th>
                            <th style={{ padding: '10px' }}>Category</th>
                            <th style={{ padding: '10px' }}>Warehouse</th>
                            <th style={{ padding: '10px' }}>Amount</th>
                            <th style={{ padding: '10px' }}>Note</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredExpenses?.map(e => (
                            <tr key={e.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(e.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{e.reference}</td>
                                <td style={{ padding: '10px' }}>{categories?.find(c => c.id === e.category_id)?.name || e.category_id}</td>
                                <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id === e.warehouse_id)?.name || '-'}</td>
                                <td style={{ padding: '10px' }}>{formatPrice(e.amount)}</td>
                                <td style={{ padding: '10px' }}>{e.note}</td>
                                <td style={{ padding: '10px' }}>
                                    {hasPermission('expenses_delete') && (
                                        <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(e.id)}><FaTrash /></button>
                                    )}
                                </td>
                            </tr>
                        ))}
                        {filteredExpenses?.length === 0 && <tr><td colSpan="7" style={{ padding: '20px', textAlign: 'center' }}>No expenses found.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Expenses;
