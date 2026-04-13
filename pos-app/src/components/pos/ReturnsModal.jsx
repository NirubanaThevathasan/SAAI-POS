import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';
import { FaTrash, FaPlus, FaUndo } from 'react-icons/fa';

const ReturnsModal = ({ onClose }) => {
    const { formatPrice, settings } = useSettings();
    const sales = useLiveQuery(() => db.sales.orderBy('date').reverse().limit(50).toArray());
    const products = useLiveQuery(() => db.products.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());
    const customers = useLiveQuery(() => db.customers.toArray());

    const [formData, setFormData] = useState({
        transaction_id: '',
        warehouse_id: settings.default_warehouse || 1,
        items: [],
        note: ''
    });

    const [productSearch, setProductSearch] = useState('');

    const handleTransactionChange = (txId) => {
        const sale = sales.find(s => s.id === parseInt(txId));
        if (sale) {
            setFormData({
                ...formData,
                transaction_id: txId,
                warehouse_id: sale.warehouse_id || settings.default_warehouse || 1,
                items: sale.items.map(i => ({ ...i, quantity: 0, maxQty: i.quantity })) // Load items but start at 0 return
            });
        }
    };

    const updateItemQty = (id, val) => {
        setFormData({
            ...formData,
            items: formData.items.map(i => i.id === id ? { ...i, quantity: Math.min(i.maxQty, parseInt(val) || 0) } : i)
        });
    };

    const handleSave = async () => {
        const itemsToReturn = formData.items.filter(i => i.quantity > 0);
        if (itemsToReturn.length === 0) return alert('Select items and quantities to return');

        const totalRefund = itemsToReturn.reduce((sum, i) => sum + (i.price * i.quantity), 0);
        const whId = parseInt(formData.warehouse_id);

        const returnData = {
            type: 'sale',
            transaction_id: parseInt(formData.transaction_id),
            warehouse_id: whId,
            customer_id: sales.find(s => s.id === parseInt(formData.transaction_id))?.customer_id || null,
            reference_no: `POS-RET-${Date.now()}`,
            date: new Date(),
            items: itemsToReturn,
            totalRefund,
            status: 'completed',
            staff_note: formData.note
        };

        await db.returns.add(returnData);

        // Adjust Stock
        for (const item of itemsToReturn) {
            // 1. Warehouse Stock
            const existingStock = await db.stock
                .where('product_id').equals(item.id)
                .filter(s => s.warehouse_id === whId)
                .first();

            if (existingStock) {
                await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) + item.quantity });
            } else {
                await db.stock.add({
                    product_id: item.id,
                    variant_id: 0,
                    warehouse_id: whId,
                    quantity: item.quantity
                });
            }

            // 2. Global Sync
            const product = await db.products.get(item.id);
            if (product) {
                const allWhStocks = await db.stock.where('product_id').equals(item.id).toArray();
                const totalWhStock = allWhStocks.reduce((sum, s) => sum + s.quantity, 0);
                await db.products.update(item.id, { stock_quantity: totalWhStock });
            }
        }

        alert('Return Processed Successfully!');
        onClose();
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1100
        }}>
            <div className="modal-content card" style={{ width: '700px', maxHeight: '90vh', overflowY: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                    <h3><FaUndo /> Process Sale Return</h3>
                    <button className="btn btn-secondary" onClick={onClose}>Close</button>
                </div>

                <div className="form-group mb-3">
                    <label>Select Original Sale</label>
                    <select className="form-control" value={formData.transaction_id} onChange={e => handleTransactionChange(e.target.value)}>
                        <option value="">Select a recent sale...</option>
                        {sales?.map(s => (
                            <option key={s.id} value={s.id}>
                                #{s.id} - {new Date(s.date).toLocaleDateString()} - {customers?.find(c => c.id === s.customer_id)?.name || 'Walk-in'} ({formatPrice(s.total)})
                            </option>
                        ))}
                    </select>
                </div>

                {formData.transaction_id && (
                    <>
                        <div className="mb-3">
                            <label>Restock Warehouse</label>
                            <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })}>
                                {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                            </select>
                        </div>

                        <table style={{ width: '100%', marginBottom: '15px' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left' }}>
                                    <th>Item</th>
                                    <th>Price</th>
                                    <th>Sold Qty</th>
                                    <th>Return Qty</th>
                                    <th>Refund</th>
                                </tr>
                            </thead>
                            <tbody>
                                {formData.items.map(item => (
                                    <tr key={item.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                        <td style={{ padding: '8px 0' }}>{item.name}</td>
                                        <td>{formatPrice(item.price)}</td>
                                        <td>{item.maxQty}</td>
                                        <td>
                                            <input
                                                type="number"
                                                className="form-control"
                                                style={{ width: '70px' }}
                                                value={item.quantity}
                                                max={item.maxQty}
                                                min="0"
                                                onChange={e => updateItemQty(item.id, e.target.value)}
                                            />
                                        </td>
                                        <td>{formatPrice(item.price * item.quantity)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <div style={{ marginBottom: '15px' }}>
                            <label>Staff Note</label>
                            <textarea className="form-control" value={formData.note} onChange={e => setFormData({ ...formData, note: e.target.value })} rows="2"></textarea>
                        </div>

                        <div className="card p-3 mb-3" style={{ background: '#f8f9fa', textAlign: 'right' }}>
                            <h4>Total Refund: <span style={{ color: 'var(--primary-color)' }}>{formatPrice(formData.items.reduce((sum, i) => sum + (i.price * i.quantity), 0))}</span></h4>
                        </div>

                        <button className="btn btn-primary w-100" onClick={handleSave}>Finalize Return & Refund</button>
                    </>
                )}
            </div>
        </div>
    );
};

export default ReturnsModal;
