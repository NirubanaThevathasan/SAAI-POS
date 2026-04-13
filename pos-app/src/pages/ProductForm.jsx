import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';

const ProductForm = ({ onClose, initialData }) => {
    const categories = useLiveQuery(() => db.categories.toArray());
    const brands = useLiveQuery(() => db.brands.toArray());
    const units = useLiveQuery(() => db.units.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [type, setType] = useState(initialData?.type || 'standard');

    // Core Data
    const [formData, setFormData] = useState({
        name: initialData?.name || '',
        code: initialData?.code || '',
        category_id: initialData?.category_id || '',
        brand_id: initialData?.brand_id || '',
        unit_id: initialData?.unit_id || '',
        cost: initialData?.cost || '',
        price: initialData?.price || '',
        description: initialData?.description || '',
        alert_quantity: initialData?.alert_quantity || 5,
        tax_method: initialData?.tax_method || 'exclusive',
        image: initialData?.image || '',
        warehouse_id: initialData?.warehouse_id || 1, // Default to 1 (Main)
        expiry_date: initialData?.expiry_date || '',
        opening_stock: '' // New Field
    });

    // Variants Data
    const [variants, setVariants] = useState([]);
    const [newVariant, setNewVariant] = useState({ name: '', code: '', price: '', cost: '' });

    // Batches Data
    const [batches, setBatches] = useState([]);
    const [newBatch, setNewBatch] = useState({ batch_no: '', expiry_date: '' });

    const handleAddVariant = () => {
        setVariants([...variants, newVariant]);
        setNewVariant({ name: '', code: '', price: '', cost: '' });
    };

    const handleAddBatch = () => {
        setBatches([...batches, newBatch]);
        setNewBatch({ batch_no: '', expiry_date: '' });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const productData = {
                name: formData.name,
                code: formData.code,
                category_id: formData.category_id,
                brand_id: formData.brand_id,
                unit_id: formData.unit_id,
                type: type,
                description: formData.description,
                alert_quantity: formData.alert_quantity !== '' ? parseInt(formData.alert_quantity) : 5,
                tax_method: formData.tax_method,
                image: formData.image,
                warehouse_id: formData.warehouse_id,
                price: parseFloat(formData.price) || 0,
                cost: parseFloat(formData.cost) || 0,
                expiry_date: formData.expiry_date,
                // Do not save opening_stock to products table directly, it's transactional
            };

            let productId;
            if (initialData?.id) {
                await db.products.update(initialData.id, productData);
                productId = initialData.id;
            } else {
                productId = await db.products.add(productData);

                // HANDLE OPENING STOCK FOR NEW PRODUCTS
                if (type === 'standard' && formData.opening_stock && formData.warehouse_id) {
                    const qty = parseInt(formData.opening_stock);
                    if (qty > 0) {
                        // 1. Add Stock Record
                        await db.stock.add({
                            product_id: productId,
                            variant_id: 0,
                            warehouse_id: parseInt(formData.warehouse_id),
                            quantity: qty
                        });

                        // 2. Update Global Stock
                        await db.products.update(productId, { stock_quantity: qty });

                        // 3. Optional: Add Adjustment Record for traceability
                        await db.adjustments.add({
                            reference_no: 'OPENING-STOCK',
                            warehouse_id: parseInt(formData.warehouse_id),
                            date: new Date(),
                            type: 'addition',
                            note: 'Initial Opening Stock',
                            items: [{ product_id: productId, name: formData.name, quantity: qty }]
                        });
                    }
                }
            }

            // 2. Add Variants if Variable
            if (type === 'variable') {
                const variantsToAdd = variants.map(v => ({
                    product_id: productId,
                    name: v.name,
                    code: v.code,
                    price: parseFloat(v.price),
                    cost: parseFloat(v.cost)
                }));
                await db.variants.bulkAdd(variantsToAdd);
            }

            // 3. Add Batches if Standard (Simple batch tracking for main product)
            // Note: complex batch tracking for variants would need variant_id linkage
            if (type === 'standard' && batches.length > 0) {
                const batchesToAdd = batches.map(b => ({
                    product_id: productId,
                    batch_no: b.batch_no,
                    expiry_date: b.expiry_date
                }));
                await db.batches.bulkAdd(batchesToAdd);
            }

            onClose();
        } catch (error) {
            console.error(error);
            alert('Error adding product');
        }
    };

    return (
        <div className="card mb-3" style={{ border: '2px solid #2980b9' }}>
            <h3>Add New Product</h3>
            <div style={{ marginBottom: '15px' }}>
                <label style={{ marginRight: '10px' }}>Product Type:</label>
                <select value={type} onChange={e => setType(e.target.value)} style={{ padding: '5px' }}>
                    <option value="standard">Standard</option>
                    <option value="variable">Variable (Sizes/Colors)</option>
                    <option value="service">Service</option>
                </select>
            </div>

            <form onSubmit={handleSubmit}>
                {/* Common Fields */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '15px' }}>
                    <div><label>Name</label><input className="form-control" required value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} style={{ width: '100%' }} /></div>
                    <div><label>Code (Barcode)</label><input className="form-control" required value={formData.code} onChange={e => setFormData({ ...formData, code: e.target.value })} style={{ width: '100%' }} /></div>
                    <div>
                        <label>Category</label>
                        <select className="form-control" value={formData.category_id} onChange={e => setFormData({ ...formData, category_id: e.target.value })} style={{ width: '100%' }}>
                            <option value="">Select Category</option>
                            {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                        </select>
                    </div>
                    <div>
                        <label>Brand</label>
                        <select className="form-control" value={formData.brand_id} onChange={e => setFormData({ ...formData, brand_id: e.target.value })} style={{ width: '100%' }}>
                            <option value="">Select Brand</option>
                            {brands?.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                        </select>
                    </div>
                    <div>
                        <label>Unit</label>
                        <select className="form-control" value={formData.unit_id} onChange={e => setFormData({ ...formData, unit_id: e.target.value })} style={{ width: '100%' }}>
                            <option value="">Select Unit</option>
                            {units?.map(u => <option key={u.id} value={u.code}>{u.name}</option>)}
                        </select>
                    </div>
                    <div>
                        <label>Warehouse</label>
                        <select className="form-control" value={formData.warehouse_id} onChange={e => setFormData({ ...formData, warehouse_id: e.target.value })} style={{ width: '100%' }}>
                            <option value="">Select Warehouse</option>
                            {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                            {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                        </select>
                    </div>
                </div>

                {/* Opening Stock Field (Only for New Standard Products) */}
                {!initialData?.id && type === 'standard' && (
                    <div className="mb-3" style={{ padding: '15px', background: '#e0f2f1', borderRadius: '4px' }}>
                        <label style={{ fontWeight: 'bold' }}>Opening Stock</label>
                        <p style={{ fontSize: '0.85rem', color: '#666', margin: '0 0 5px 0' }}>Immediately add this quantity to the selected warehouse.</p>
                        <input
                            type="number"
                            min="0"
                            className="form-control"
                            placeholder="0"
                            value={formData.opening_stock}
                            onChange={e => setFormData({ ...formData, opening_stock: e.target.value })}
                        />
                    </div>
                )}

                {/* Additional Details */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '15px' }}>
                    <div>
                        <label>Alert Quantity</label>
                        <input type="number" step="1" min="0" className="form-control" value={formData.alert_quantity} onChange={e => setFormData({ ...formData, alert_quantity: e.target.value })} />
                    </div>
                    <div>
                        <label>Tax Method</label>
                        <select className="form-control" value={formData.tax_method} onChange={e => setFormData({ ...formData, tax_method: e.target.value })}>
                            <option value="exclusive">Exclusive</option>
                            <option value="inclusive">Inclusive</option>
                        </select>
                    </div>
                    <div>
                        <label>Image URL</label>
                        <input type="text" className="form-control" value={formData.image} onChange={e => setFormData({ ...formData, image: e.target.value })} placeholder="http://..." />
                    </div>
                </div>

                <div className="mb-3">
                    <label>Description</label>
                    <textarea className="form-control" rows="3" value={formData.description} onChange={e => setFormData({ ...formData, description: e.target.value })} style={{ width: '100%' }}></textarea>
                </div>

                {/* Standard Specific Fields */}
                {type === 'standard' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', padding: '15px', background: '#f8f9fa' }}>
                        <div><label>Cost *</label><input type="number" min="0" step="0.01" className="form-control" required value={formData.cost} onChange={e => setFormData({ ...formData, cost: e.target.value })} style={{ width: '100%' }} /></div>
                        <div><label>Price *</label><input type="number" min="0" step="0.01" className="form-control" required value={formData.price} onChange={e => setFormData({ ...formData, price: e.target.value })} style={{ width: '100%' }} /></div>
                        <div><label>Expiry Date</label><input type="date" className="form-control" value={formData.expiry_date} onChange={e => setFormData({ ...formData, expiry_date: e.target.value })} style={{ width: '100%' }} /></div>

                        {/* Batch Entry */}
                        <div style={{ gridColumn: '1 / -1', marginTop: '10px' }}>
                            <h5>Batches (Optional)</h5>
                            {batches.map((b, i) => <div key={i}><small>{b.batch_no} - {b.expiry_date} ({b.quantity})</small></div>)}
                            <div style={{ display: 'flex', gap: '5px', marginTop: '5px' }}>
                                <input placeholder="Batch No" value={newBatch.batch_no} onChange={e => setNewBatch({ ...newBatch, batch_no: e.target.value })} />
                                <input type="date" value={newBatch.expiry_date} onChange={e => setNewBatch({ ...newBatch, expiry_date: e.target.value })} />
                                <button type="button" onClick={handleAddBatch}>+ Batch</button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Variable Specific Fields */}
                {type === 'variable' && (
                    <div style={{ padding: '15px', background: '#e8f6f3' }}>
                        <h4>Variants</h4>
                        <table style={{ width: '100%', marginBottom: '10px' }}>
                            <thead><tr><th>Name (e.g. Red-S)</th><th>Code</th><th>Price</th></tr></thead>
                            <tbody>
                                {variants.map((v, i) => (
                                    <tr key={i}><td>{v.name}</td><td>{v.code}</td><td>{v.price}</td></tr>
                                ))}
                                <tr>
                                    <td><input placeholder="Name" value={newVariant.name} onChange={e => setNewVariant({ ...newVariant, name: e.target.value })} style={{ width: '100%' }} /></td>
                                    <td><input placeholder="Code" value={newVariant.code} onChange={e => setNewVariant({ ...newVariant, code: e.target.value })} style={{ width: '100%' }} /></td>
                                    <td><input type="number" min="0" step="0.01" placeholder="Price" value={newVariant.price} onChange={e => setNewVariant({ ...newVariant, price: e.target.value })} style={{ width: '100%' }} /></td>
                                </tr>
                            </tbody>
                        </table>
                        <button type="button" onClick={handleAddVariant} className="btn-sm">+ Add Variant</button>
                    </div>
                )}

                <div style={{ marginTop: '20px' }}>
                    <button type="submit" className="btn btn-primary">Save Product</button>
                    <button type="button" className="btn" onClick={onClose} style={{ marginLeft: '10px' }}>Cancel</button>
                </div>
            </form>
        </div>
    );
};

export default ProductForm;
