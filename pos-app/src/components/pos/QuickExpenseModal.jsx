import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';

const QuickExpenseModal = ({ onClose }) => {
    const { currency } = useSettings();
    const categories = useLiveQuery(() => db.expense_categories.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [formData, setFormData] = useState({
        reference: `EXP-${Date.now()}`,
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
        alert('Expense Added!');
        onClose();
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)',
            display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
        }}>
            <div className="modal-content card" style={{ width: '500px', padding: '20px' }}>
                <h3>Quick Expense</h3>
                <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div>
                        <label>Amount</label>
                        <div style={{ position: 'relative' }}>
                            <span style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#64748b', fontWeight: 'bold' }}>{currency.symbol}</span>
                            <input type="number" className="form-control" autoFocus required value={formData.amount} onChange={e => setFormData({ ...formData, amount: e.target.value })} style={{ width: '100%', padding: '8px 8px 8px 35px' }} />
                        </div>
                    </div>
                    <div>
                        <label>Category</label>
                        <select className="form-control" required value={formData.category_id} onChange={e => setFormData({ ...formData, category_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                            <option value="">Select...</option>
                            {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                        </select>
                    </div>
                    <div>
                        <label>Reference</label>
                        <input type="text" className="form-control" value={formData.reference} onChange={e => setFormData({ ...formData, reference: e.target.value })} style={{ width: '100%', padding: '8px' }} />
                    </div>
                    <div>
                        <label>Warehouse</label>
                        <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                            <option value="">Optional...</option>
                            {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                        </select>
                    </div>
                    <div style={{ gridColumn: '1 / -1' }}>
                        <label>Note</label>
                        <input type="text" className="form-control" value={formData.note} onChange={e => setFormData({ ...formData, note: e.target.value })} style={{ width: '100%', padding: '8px' }} />
                    </div>
                    <div style={{ gridColumn: '1 / -1', display: 'flex', gap: '10px', marginTop: '10px' }}>
                        <button type="submit" className="btn btn-primary" style={{ flex: 1 }}>Save</button>
                        <button type="button" className="btn btn-secondary" onClick={onClose} style={{ flex: 1 }}>Cancel</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default QuickExpenseModal;
