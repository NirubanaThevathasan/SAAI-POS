import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { FaPlus, FaCheck } from 'react-icons/fa';

const StockCounts = () => {
    const counts = useLiveQuery(() => db.stock_counts.orderBy('date').reverse().toArray());
    const products = useLiveQuery(() => db.products.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [showForm, setShowForm] = useState(false);
    const [formData, setFormData] = useState({
        reference: '', warehouse_id: '', type: 'full'
    });
    const [countData, setCountData] = useState([]); // { product_id, expected, counted }

    const handleLoadProducts = async () => {
        if (!formData.warehouse_id) return alert('Select Warehouse');

        const whId = parseInt(formData.warehouse_id);
        const allProducts = await db.products.toArray();
        const allStock = await db.stock.where('warehouse_id').equals(whId).toArray();

        const initialCounts = allProducts.map(p => {
            const whStock = allStock.find(s => s.product_id === p.id);
            const expected = whStock ? whStock.quantity : 0;
            return {
                product: p,
                expected: expected,
                counted: expected // Default to expected
            };
        });
        setCountData(initialCounts);
    };

    const handleCountChange = (index, value) => {
        const newData = [...countData];
        newData[index].counted = parseInt(value);
        setCountData(newData);
    };

    const handleFinalize = async () => {
        if (!window.confirm('This will overwrite current stock levels for this warehouse. Continue?')) return;

        const whId = parseInt(formData.warehouse_id);

        try {
            await db.transaction('rw', [db.stock_counts, db.stock, db.products], async () => {
                // 1. Save Count Record
                await db.stock_counts.add({
                    ...formData,
                    warehouse_id: whId,
                    date: new Date(),
                    items: countData.map(c => ({ product_id: c.product.id, expected: c.expected, counted: c.counted }))
                });

                // 2. Update Stock for each item
                for (const item of countData) {
                    const countedValue = parseInt(item.counted) || 0;

                    // Update/Add Warehouse Stock
                    const existingStock = await db.stock
                        .where('[product_id+variant_id+warehouse_id]')
                        .equals([item.product.id, 0, whId])
                        .first();

                    if (existingStock) {
                        await db.stock.update(existingStock.id, { quantity: countedValue });
                    } else if (countedValue !== 0) {
                        await db.stock.add({
                            product_id: item.product.id,
                            variant_id: 0,
                            warehouse_id: whId,
                            quantity: countedValue
                        });
                    }
                }

                // 3. Re-sync ALL products' global stock sums
                // This is more reliable than per-product await in loop
                const allStockEntries = await db.stock.toArray();
                const stockByProduct = {};
                allStockEntries.forEach(s => {
                    stockByProduct[s.product_id] = (stockByProduct[s.product_id] || 0) + s.quantity;
                });

                const allProducts = await db.products.toArray();
                for (const product of allProducts) {
                    const newGlobalStock = stockByProduct[product.id] || 0;
                    if (product.stock_quantity !== newGlobalStock) {
                        await db.products.update(product.id, { stock_quantity: newGlobalStock });
                    }
                }
            });

            setShowForm(false);
            setCountData([]);
            setFormData({ reference: '', warehouse_id: '', type: 'full' });
            alert('Stock Count Finalized & Warehouse Stock Synced');
        } catch (error) {
            console.error('Stock count finalization failed:', error);
            alert('Error updating stock: ' + error.message);
        }
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3>Stock Counts</h3>
                <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}><FaPlus /> Perform Stock Count</button>
            </div>

            {showForm && (
                <div className="card mb-3" style={{ border: '1px solid #9b59b6' }}>
                    <h4>New Stock Count</h4>
                    <div style={{ display: 'flex', gap: '15px', marginBottom: '15px' }}>
                        <select className="form-control" onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })}>
                            <option value="">Select Warehouse...</option>
                            {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                        </select>
                        <select className="form-control" value={formData.type} onChange={e => setFormData({ ...formData, type: e.target.value })}>
                            <option value="full">Full Count</option>
                            <option value="partial">Partial (Not Implemented)</option>
                        </select>
                        <input className="form-control" placeholder="Reference" onChange={e => setFormData({ ...formData, reference: e.target.value })} />
                        <button className="btn btn-info" onClick={handleLoadProducts}>Load Products</button>
                    </div>

                    {countData.length > 0 && (
                        <>
                            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                                <table style={{ width: '100%' }}>
                                    <thead>
                                        <tr style={{ position: 'sticky', top: 0, background: '#fff' }}><th>Product</th><th>Expected</th><th>Counted</th><th>Diff</th></tr>
                                    </thead>
                                    <tbody>
                                        {countData.map((item, i) => (
                                            <tr key={i} style={{ background: item.expected !== item.counted ? '#fff3cd' : 'transparent' }}>
                                                <td>{item.product.name}</td>
                                                <td>{item.expected}</td>
                                                <td>
                                                    <input type="number" style={{ width: '80px' }} value={item.counted} onChange={e => handleCountChange(i, e.target.value)} />
                                                </td>
                                                <td style={{ color: item.counted - item.expected < 0 ? 'red' : 'green' }}>
                                                    {item.counted - item.expected}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <button className="btn btn-success" style={{ marginTop: '15px' }} onClick={handleFinalize}><FaCheck /> Finalize & Update Stock</button>
                        </>
                    )}
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
                        </tr>
                    </thead>
                    <tbody>
                        {counts?.map(c => (
                            <tr key={c.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(c.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{c.reference}</td>
                                <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id == c.warehouse_id)?.name}</td>
                                <td style={{ padding: '10px' }}>{c.type}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default StockCounts;
