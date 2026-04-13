import React, { useState } from 'react';

const CustomerModal = ({ onClose, onAdd }) => {
    const [formData, setFormData] = useState({
        name: '',
        phone: '',
        email: '',
        city: ''
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        onAdd(formData);
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
        }}>
            <div className="modal-content card" style={{ padding: '20px', width: '400px', background: 'white' }}>
                <h3>Add New Customer</h3>
                <form onSubmit={handleSubmit}>
                    <div className="form-group mb-2">
                        <label>Name</label>
                        <input type="text" className="form-control" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} required />
                    </div>
                    <div className="form-group mb-2">
                        <label>Phone</label>
                        <input type="text" className="form-control" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} />
                    </div>
                    <div className="form-group mb-2">
                        <label>Email</label>
                        <input type="email" className="form-control" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} />
                    </div>
                    <div className="form-group mb-3">
                        <label>City</label>
                        <input type="text" className="form-control" value={formData.city} onChange={e => setFormData({ ...formData, city: e.target.value })} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                        <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
                        <button type="submit" className="btn btn-primary">Save & Select</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default CustomerModal;
