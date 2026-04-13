import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { FaPlus, FaTrash } from 'react-icons/fa';

const QuantityAdjustments = () => {
    const adjustments = useLiveQuery(() => db.adjustments.orderBy('date').reverse().toArray());
    const products = useLiveQuery(() => db.products.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    // Form State
    const [showForm, setShowForm] = useState(false);
    const [formData, setFormData] = useState({
        reference_no: '',
        warehouse_id: '',
        type: 'addition', // or subtraction
        note: ''
    });
    const [items, setItems] = useState([]); // { product_id, quantity }

    // Helper to add item to adjustment
    const addItem = (productId) => {
        const product = products.find(p => p.id === parseInt(productId));
        if (!product) return;
        setItems([...items, { product: product, quantity: 1 }]);
    };

    const updateItemQty = (index, qty) => {
        const newItems = [...items];
        newItems[index].quantity = parseInt(qty);
        setItems(newItems);
    }

    const removeItem = (index) => {
        setItems(items.filter((_, i) => i !== index));
    };

    const handleSubmit = async () => {
        if (!formData.warehouse_id || items.length === 0) return alert('Select warehouse and items');

        // 1. Save Adjustment Record
        await db.adjustments.add({
            ...formData,
            date: new Date(),
            items: items.map(i => ({ product_id: i.product.id, name: i.product.name, quantity: i.quantity }))
        });

        // 2. Update Product Stock
        for (const item of items) {
            const product = await db.products.get(item.product.id);
            const multiplier = formData.type === 'addition' ? 1 : -1;

            // Update Global Stock
            let newGlobalStock = (product.stock_quantity || 0) + (item.quantity * multiplier);
            await db.products.update(item.product.id, { stock_quantity: newGlobalStock });

            // Update Warehouse-Specific Stock
            const warehouseId = parseInt(formData.warehouse_id);
            const existingStock = await db.stock
                .where('product_id').equals(item.product.id)
                .filter(s => s.warehouse_id === warehouseId)
                .first();

            if (existingStock) {
                await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) + (item.quantity * multiplier) });
            } else {
                await db.stock.add({
                    product_id: item.product.id,
                    variant_id: 0,
                    warehouse_id: warehouseId,
                    quantity: item.quantity * multiplier
                });
            }
        }

        setShowForm(false);
        setFormData({ reference_no: '', warehouse_id: '', type: 'addition', note: '' });
        setItems([]);
        alert('Adjustment Saved & Stock Updated');
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3>Quantity Adjustments</h3>
                <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}><FaPlus /> Add Adjustment</button>
            </div>

            {showForm && (
                <div className="card mb-3" style={{ border: '1px solid #e67e22' }}>
                    <h4>New Adjustment</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '15px' }}>
                        <div>
                            <label>Warehouse</label>
                            <select className="form-control" onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })}>
                                <option value="">Select...</option>
                                {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                            </select>
                        </div>
                        <div>
                            <label>Type</label>
                            <select className="form-control" value={formData.type} onChange={e => setFormData({ ...formData, type: e.target.value })}>
                                <option value="addition">Addition (+)</option>
                                <option value="subtraction">Subtraction (-)</option>
                            </select>
                        </div>
                        <div>
                            <label>Reference</label>
                            <input className="form-control" placeholder="e.g. ADJ-001" onChange={e => setFormData({ ...formData, reference_no: e.target.value })} />
                        </div>
                    </div>

                    <div className="form-group mb-3">
                        <label>Add Product</label>
                        <select className="form-control" onChange={e => addItem(e.target.value)} value="">
                            <option value="">Search/Select Product...</option>
                            {products?.map(p => <option key={p.id} value={p.id}>{p.name} ({p.code})</option>)}
                        </select>
                    </div>

                    <table style={{ width: '100%', marginBottom: '10px' }}>
                        <thead><tr><th>Product</th><th>Quantity</th><th>Action</th></tr></thead>
                        <tbody>
                            {items.map((item, i) => (
                                <tr key={i}>
                                    <td>{item.product.name}</td>
                                    <td>
                                        <input type="number" style={{ width: '80px' }} value={item.quantity} onChange={e => updateItemQty(i, e.target.value)} />
                                    </td>
                                    <td><button className="btn-icon" onClick={() => removeItem(i)} style={{ color: 'red' }}><FaTrash /></button></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    <button className="btn btn-success" onClick={handleSubmit}>Submit Adjustment</button>
                </div>
            )}

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Ref</th>
                            <th style={{ padding: '10px' }}>Warehouse</th>
                            <th style={{ padding: '10px' }}>Type</th>
                            <th style={{ padding: '10px' }}>Items</th>
                        </tr>
                    </thead>
                    <tbody>
                        {adjustments?.map(adj => (
                            <tr key={adj.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(adj.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{adj.reference_no}</td>
                                <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id == adj.warehouse_id)?.name}</td>
                                <td style={{ padding: '10px' }}>
                                    <span style={{
                                        padding: '2px 8px', borderRadius: '4px',
                                        background: adj.type === 'addition' ? '#d4edda' : '#f8d7da',
                                        color: adj.type === 'addition' ? '#155724' : '#721c24'
                                    }}>
                                        {adj.type}
                                    </span>
                                </td>
                                <td style={{ padding: '10px' }}>{adj.items?.length} items</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default QuantityAdjustments;
