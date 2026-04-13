import React, { useState } from 'react';
import { useSettings } from '../../context/SettingsContext';

const ManualItemModal = ({ onClose, onAdd }) => {
    const { currency } = useSettings();
    const [name, setName] = useState('Custom Item');
    const [price, setPrice] = useState(0);
    const [cost, setCost] = useState(0);
    const [quantity, setQuantity] = useState(1);

    const handleSubmit = (e) => {
        e.preventDefault();
        const qty = parseInt(quantity);
        if (isNaN(qty) || qty <= 0) return alert('Quantity must be greater than 0');

        onAdd({
            id: 'manual_' + Date.now(),
            name,
            price: parseFloat(price),
            cost: parseFloat(cost) || 0,
            stock_quantity: 9999, // Unlimited for manual items
            isManual: true
        }, qty);
        onClose();
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
        }}>
            <div className="modal-content card" style={{ padding: '20px', width: '400px', background: 'white' }}>
                <h3>Add Manual Item</h3>
                <form onSubmit={handleSubmit}>
                    <div className="form-group mb-3">
                        <label>Item Name</label>
                        <input type="text" className="form-control" value={name} onChange={e => setName(e.target.value)} required />
                    </div>
                    <div className="form-group mb-3">
                        <label>Cost Price ({currency.symbol})</label>
                        <div style={{ position: 'relative' }}>
                            <span style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }}>{currency.symbol}</span>
                            <input type="number" step="0.01" className="form-control" value={cost} onChange={e => setCost(e.target.value)} required style={{ paddingLeft: '40px' }} />
                        </div>
                    </div>
                    <div className="form-group mb-3">
                        <label>Sale Price ({currency.symbol})</label>
                        <div style={{ position: 'relative' }}>
                            <span style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }}>{currency.symbol}</span>
                            <input type="number" step="0.01" className="form-control" value={price} onChange={e => setPrice(e.target.value)} required style={{ paddingLeft: '40px' }} />
                        </div>
                    </div>
                    <div className="form-group mb-3">
                        <label>Quantity</label>
                        <input type="number" className="form-control" value={quantity} onChange={e => setQuantity(e.target.value)} min="1" required />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                        <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
                        <button type="submit" className="btn btn-primary">Add to Cart</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ManualItemModal;
