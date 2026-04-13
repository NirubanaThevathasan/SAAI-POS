import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';

const ManualSale = () => {
    const { formatPrice } = useSettings();
    const products = useLiveQuery(() => db.products.toArray());
    const customers = useLiveQuery(() => db.customers.toArray());

    const [formData, setFormData] = useState({
        customer_id: '',
        date: new Date().toISOString().slice(0, 10),
        status: 'completed',
        paymentMethod: 'Cash',
        note: ''
    });

    const [items, setItems] = useState([]); // { product, quantity, price }

    const warehouseStock = useLiveQuery(async () => {
        const allSettings = await db.settings.toArray();
        const settingsObj = {};
        allSettings.forEach(s => settingsObj[s.key] = s.value);
        const whId = parseInt(settingsObj.default_warehouse || 1);
        return await db.stock.where('warehouse_id').equals(whId).toArray();
    }, []);

    const getStockForProduct = (productId) => {
        const s = warehouseStock?.find(s => s.product_id === productId);
        return s ? s.quantity : 0;
    };

    const addItem = (productId) => {
        const product = products.find(p => p.id === parseInt(productId));
        if (!product) return;

        const available = getStockForProduct(product.id);
        const existingItem = items.find(i => i.product.id === product.id);
        const currentQty = existingItem ? existingItem.quantity : 0;

        if (currentQty + 1 > available) {
            return alert(`Insufficient stock for "${product.name}"! Only ${available} available.`);
        }

        if (existingItem) {
            setItems(items.map(i =>
                i.product.id === product.id ? { ...i, quantity: i.quantity + 1 } : i
            ));
        } else {
            setItems([...items, { product, quantity: 1, price: product.price }]);
        }
    };

    const removeItem = (index) => {
        setItems(items.filter((_, i) => i !== index));
    };

    const updateItem = (index, field, value) => {
        const newItems = [...items];
        const val = parseFloat(value) || 0;

        if (field === 'quantity') {
            const available = getStockForProduct(newItems[index].product.id);
            if (val > available) {
                alert(`Insufficient stock! Max available: ${available}`);
                return;
            }
        }

        newItems[index][field] = val;
        setItems(newItems);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (items.length === 0) return alert('Please add at least one product.');

        const hasInvalidQty = items.some(item => item.quantity <= 0);
        if (hasInvalidQty) return alert('All items must have a quantity greater than 0.');

        // Re-validate stock one last time before saving
        const allSettings = await db.settings.toArray();
        const settingsObj = {};
        allSettings.forEach(s => settingsObj[s.key] = s.value);
        const whId = parseInt(settingsObj.default_warehouse || 1);

        for (const item of items) {
            const stockEntry = await db.stock
                .where('[product_id+variant_id+warehouse_id]')
                .equals([item.product.id, 0, whId])
                .first();
            const available = stockEntry ? stockEntry.quantity : 0;
            if (item.quantity > available) {
                return alert(`Stock changed for "${item.product.name}"! Available: ${available}.`);
            }
        }

        const total = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);

        // Prepare proper sale object matching POS structure
        const saleData = {
            ...formData,
            customer_id: formData.customer_id ? parseInt(formData.customer_id) : null,
            warehouse_id: whId,
            items: items.map(i => ({
                ...i.product,
                id: i.product.id, // Ensure ID is preserved
                quantity: i.quantity,
                price: parseFloat(i.price),
                cost: i.product.cost || 0
            })),
            subtotal: total,
            tax: 0,
            discount: 0,
            total: total,
            received: total, // Assume paid in full for manual entry
            balance: 0,
            source: 'manual',
            date: new Date(formData.date)
        };

        await db.sales.add(saleData);

        // 3. Update stock levels
        for (const item of items) {
            // Update Warehouse Stock
            const existingStock = await db.stock
                .where('[product_id+variant_id+warehouse_id]')
                .equals([item.product.id, 0, whId])
                .first();

            if (existingStock) {
                await db.stock.update(existingStock.id, { quantity: (existingStock.quantity || 0) - item.quantity });
            } else {
                await db.stock.add({
                    product_id: item.product.id,
                    variant_id: 0,
                    warehouse_id: whId,
                    quantity: -item.quantity
                });
            }

            // Global Stock Re-sync
            const product = await db.products.get(item.product.id);
            if (product) {
                const allWhStocks = await db.stock.where('product_id').equals(item.product.id).toArray();
                const totalWhStock = allWhStocks.reduce((sum, s) => sum + s.quantity, 0);
                await db.products.update(item.product.id, { stock_quantity: totalWhStock });
            }
        }

        // Import dynamically to avoid circular dependencies if any
        const AutoBackup = await import('../../services/AutoBackupService');
        if (AutoBackup && AutoBackup.triggerAutoBackup) {
            AutoBackup.triggerAutoBackup();
        }

        alert('Sale Added & Stock Updated Successfully');
        setItems([]);
        setFormData({ ...formData, note: '' }); // Reset form
    };

    return (
        <div className="card">
            <h3>Add Manual Sale (Invoice)</h3>
            <form onSubmit={handleSubmit}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px', marginBottom: '20px' }}>
                    <div>
                        <label>Customer</label>
                        <select className="form-control" onChange={e => setFormData({ ...formData, customer_id: e.target.value })}>
                            <option value="">Walk-in Customer</option>
                            {customers?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                        </select>
                    </div>
                    <div>
                        <label>Date</label>
                        <input type="date" className="form-control" value={formData.date} onChange={e => setFormData({ ...formData, date: e.target.value })} />
                    </div>
                    <div>
                        <label>Payment Method</label>
                        <select className="form-control" onChange={e => setFormData({ ...formData, paymentMethod: e.target.value })}>
                            <option>Cash</option>
                            <option>Card</option>
                            <option>UPI</option>
                        </select>
                    </div>
                </div>

                <div className="form-group mb-3">
                    <label>Add Product</label>
                    <select className="form-control" onChange={e => addItem(e.target.value)} value="">
                        <option value="">Select product to add...</option>
                        {products?.map(p => <option key={p.id} value={p.id}>{p.name} ({formatPrice(p.price)})</option>)}
                    </select>
                </div>

                {items.length > 0 && (
                    <table style={{ width: '100%', marginBottom: '20px' }}>
                        <thead><tr><th>Product</th><th>Price</th><th>Qty</th><th>Total</th><th>Action</th></tr></thead>
                        <tbody>
                            {items.map((item, i) => (
                                <tr key={i}>
                                    <td>{item.product.name}</td>
                                    <td><input type="number" style={{ width: '80px' }} value={item.price} onChange={e => updateItem(i, 'price', e.target.value)} /></td>
                                    <td><input type="number" style={{ width: '60px' }} value={item.quantity} onChange={e => updateItem(i, 'quantity', e.target.value)} /></td>
                                    <td>{formatPrice(item.price * item.quantity)}</td>
                                    <td><button type="button" onClick={() => removeItem(i)} style={{ color: 'red' }}>X</button></td>
                                </tr>
                            ))}
                            <tr>
                                <td colSpan="3" style={{ textAlign: 'right' }}><strong>Total:</strong></td>
                                <td><strong>{formatPrice(items.reduce((acc, i) => acc + (i.price * i.quantity), 0))}</strong></td>
                                <td></td>
                            </tr>
                        </tbody>
                    </table>
                )}

                <button type="submit" className="btn btn-success">Complete Sale</button>
            </form>
        </div>
    );
};

export default ManualSale;
