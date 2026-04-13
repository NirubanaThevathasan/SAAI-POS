import React, { useState, useRef } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../db';
import { FaPlus, FaTrash, FaPrint, FaEdit, FaSearch, FaCheck } from 'react-icons/fa';
import { useReactToPrint } from 'react-to-print';
import Barcode from 'react-barcode';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';

const TransferInvoice = React.forwardRef(({ transfer, warehouses }, ref) => {
    const { formatPrice } = useSettings();
    if (!transfer) return null;
    const fromWH = warehouses?.find(w => w.id === transfer.from_warehouse_id);
    const toWH = warehouses?.find(w => w.id === transfer.to_warehouse_id);

    return (
        <div ref={ref} style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h2 style={{ textAlign: 'center' }}>Stock Transfer</h2>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                <p><strong>Reference:</strong> {transfer.reference_no}</p>
                <p><strong>Date:</strong> {new Date(transfer.date).toLocaleString()}</p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div style={{ width: '45%' }}>
                    <h4>From:</h4>
                    <p>{fromWH?.name || 'Unknown'}</p>
                </div>
                <div style={{ width: '45%', textAlign: 'right' }}>
                    <h4>To:</h4>
                    <p>{toWH?.name || 'Unknown'}</p>
                </div>
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                <thead>
                    <tr style={{ borderBottom: '2px solid #000' }}>
                        <th style={{ textAlign: 'left', padding: '5px' }}>Item</th>
                        <th style={{ textAlign: 'right', padding: '5px' }}>Quantity</th>
                    </tr>
                </thead>
                <tbody>
                    {transfer.items?.map((item, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                            <td style={{ padding: '5px' }}>{item.name} ({item.code})</td>
                            <td style={{ padding: '5px', textAlign: 'right' }}>{item.quantity}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <div style={{ marginTop: '30px' }}>
                <p><strong>Status:</strong> {transfer.status}</p>
                {transfer.note && <p><strong>Note:</strong> {transfer.note}</p>}
            </div>
            <div style={{ marginTop: '30px', textAlign: 'center' }}>
                <Barcode value={transfer.reference_no} height={40} />
            </div>
        </div>
    );
});

const Transfers = () => {
    const { formatPrice } = useSettings();
    const { hasPermission } = useAuth();
    const transfers = useLiveQuery(() => db.transfers.orderBy('date').reverse().toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());
    const products = useLiveQuery(() => db.products.toArray());

    const [showForm, setShowForm] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');

    // Form State
    const [formData, setFormData] = useState({
        reference_no: '',
        from_warehouse_id: '',
        to_warehouse_id: '',
        status: 'pending', // pending, sent, completed
        note: '',
        items: []
    });

    const [productSearch, setProductSearch] = useState('');

    // Print logic
    const componentRef = useRef();
    const [printTransfer, setPrintTransfer] = useState(null);
    const handlePrint = useReactToPrint({
        contentRef: componentRef,
        onAfterPrint: () => setPrintTransfer(null)
    });

    const initiatePrint = (transfer) => {
        setPrintTransfer(transfer);
        setTimeout(handlePrint, 100);
    };

    const addItem = (product) => {
        const existing = formData.items.find(i => i.id === product.id);
        if (existing) {
            setFormData({
                ...formData,
                items: formData.items.map(i => i.id === product.id ? { ...i, quantity: i.quantity + 1 } : i)
            });
        } else {
            setFormData({
                ...formData,
                items: [...formData.items, { ...product, quantity: 1 }]
            });
        }
    };

    const removeItem = (id) => {
        setFormData({ ...formData, items: formData.items.filter(i => i.id !== id) });
    };

    const updateItemQty = (id, qty) => {
        setFormData({
            ...formData,
            items: formData.items.map(i => i.id === id ? { ...i, quantity: parseFloat(qty) } : i)
        });
    };

    const handleSave = async (e) => {
        e.preventDefault();
        if (!formData.from_warehouse_id || !formData.to_warehouse_id) return alert('Select warehouses');
        if (formData.from_warehouse_id === formData.to_warehouse_id) return alert('Source and Destination must be different');
        if (formData.items.length === 0) return alert('Add products to transfer');

        const transferData = {
            ...formData,
            reference_no: formData.reference_no || `TR-${Date.now()}`,
            date: new Date(),
            from_warehouse_id: parseInt(formData.from_warehouse_id),
            to_warehouse_id: parseInt(formData.to_warehouse_id)
        };

        await db.transfers.add(transferData);

        // If status is 'completed', move stock
        if (transferData.status === 'completed') {
            await adjustStock(transferData.items, transferData.from_warehouse_id, -1);
            await adjustStock(transferData.items, transferData.to_warehouse_id, 1);
        }

        resetForm();
    };

    const adjustStock = async (items, warehouseId, multiplier) => {
        for (const item of items) {
            const whId = parseInt(warehouseId);
            const qty = parseFloat(item.quantity) || 0;

            // 1. Find existing warehouse-specific stock record
            const existingStock = await db.stock
                .where('product_id').equals(item.id)
                .filter(s => s.warehouse_id === whId)
                .first();

            if (existingStock) {
                // Record exists — apply the delta directly
                const newWhQty = (existingStock.quantity || 0) + (qty * multiplier);
                await db.stock.update(existingStock.id, { quantity: newWhQty });
            } else {
                // No warehouse record for this product yet.
                // Products added via Opening Stock only set products.stock_quantity
                // and never create a stock table row. We must seed correctly:
                const product = await db.products.get(item.id);
                const globalQty = parseFloat(product?.stock_quantity || 0);
                const allExisting = await db.stock.where('product_id').equals(item.id).toArray();

                // If NO warehouse records exist at all → seed from global so stock isn't lost
                const seedQty = allExisting.length === 0 ? globalQty : 0;

                await db.stock.add({
                    product_id: item.id,
                    variant_id: 0,
                    warehouse_id: whId,
                    quantity: seedQty + (qty * multiplier)
                });
            }

            // 2. Global Re-sync: sum all warehouse rows → write to products.stock_quantity
            const allWhStocks = await db.stock.where('product_id').equals(item.id).toArray();
            const totalWhStock = allWhStocks.reduce((sum, s) => sum + (parseFloat(s.quantity) || 0), 0);
            await db.products.update(item.id, { stock_quantity: totalWhStock });
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (evt) => {
            const text = evt.target.result;
            const rows = text.split('\n').slice(1); // Skip header

            let count = 0;
            for (const row of rows) {
                if (!row.trim()) continue;
                // Expected format: Date,FromWarehouseCode,ToWarehouseCode,ProductCode,Quantity
                const [dateStr, fromCode, toCode, productCode, qtyStr] = row.split(',');

                const fromWH = await db.warehouses.where('code').equals(fromCode?.trim()).first();
                const toWH = await db.warehouses.where('code').equals(toCode?.trim()).first();
                const product = await db.products.where('code').equals(productCode?.trim()).first();

                if (fromWH && toWH && product) {
                    await db.transfers.add({
                        date: new Date(dateStr || new Date()),
                        reference_no: `IMP-${Date.now()}-${count}`,
                        from_warehouse_id: fromWH.id,
                        to_warehouse_id: toWH.id,
                        status: 'completed', // Auto-complete imports? Or pending? Let's say pending to be safe, or completed if user expects it. Let's do completed for bulk imports usually implies history.
                        items: [{
                            id: product.id,
                            name: product.name,
                            code: product.code,
                            quantity: parseFloat(qtyStr) || 1
                        }]
                    });
                    // If completed, adjust stock?
                    // For now, let's assume imports are historical or need manual check. 
                    // Let's set to 'completed' and adjust stock because that's usually the intent of "bulk entry".
                    await adjustStock([{ id: product.id, quantity: parseFloat(qtyStr) || 1 }], fromWH.id, -1);
                    await adjustStock([{ id: product.id, quantity: parseFloat(qtyStr) || 1 }], toWH.id, 1);
                    count++;
                }
            }
            alert(`Imported ${count} transfers!`);
            e.target.value = '';
        };
        reader.readAsText(file);
    };

    const resetForm = () => {
        setFormData({ reference_no: '', from_warehouse_id: '', to_warehouse_id: '', status: 'pending', note: '', items: [] });
        setShowForm(false);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete transfer? Stock will NOT be reverted automatically (manual adjustment required if completed).')) {
            await db.transfers.delete(id);
        }
    };

    const filteredTransfers = transfers?.filter(t => {
        const matchesSearch = searchTerm === '' || t.reference_no.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesStatus = statusFilter === 'all' || t.status === statusFilter;
        return matchesSearch && matchesStatus;
    });

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Transfers</h2>
                {hasPermission('transfers_add') && (
                    <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
                        <FaPlus /> Add Transfer
                    </button>
                )}
            </div>

            <div style={{ display: 'none' }}>
                <TransferInvoice ref={componentRef} transfer={printTransfer} warehouses={warehouses} />
            </div>

            {showForm && (
                <div className="card mb-3">
                    <h3>New Transfer</h3>
                    <form onSubmit={handleSave}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                            <div>
                                <label>Reference</label>
                                <input className="form-control" value={formData.reference_no} onChange={e => setFormData({ ...formData, reference_no: e.target.value })} placeholder="Auto-generated" />
                            </div>
                            <div>
                                <label>From Warehouse</label>
                                <select className="form-control" value={formData.from_warehouse_id} onChange={e => setFormData({ ...formData, from_warehouse_id: e.target.value })} required>
                                    <option value="">Select Source</option>
                                    {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                                </select>
                            </div>
                            <div>
                                <label>To Warehouse</label>
                                <select className="form-control" value={formData.to_warehouse_id} onChange={e => setFormData({ ...formData, to_warehouse_id: e.target.value })} required>
                                    <option value="">Select Destination</option>
                                    {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                                </select>
                            </div>
                            <div>
                                <label>Status</label>
                                <select className="form-control" value={formData.status} onChange={e => setFormData({ ...formData, status: e.target.value })}>
                                    <option value="pending">Pending</option>
                                    <option value="sent">Sent</option>
                                    <option value="completed">Completed</option>
                                </select>
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', color: 'var(--primary-color)' }}>
                                    Estimated Transfer Value: {formatPrice(formData.items.reduce((sum, i) => sum + (i.cost * i.quantity || 0), 0))}
                                </label>
                            </div>
                        </div>

                        <div style={{ marginBottom: '15px', padding: '10px', background: '#f9f9f9', border: '1px solid #eee' }}>
                            <h4>Add Items</h4>
                            <input
                                className="form-control"
                                placeholder="Search products..."
                                value={productSearch}
                                onChange={e => setProductSearch(e.target.value)}
                            />
                            {productSearch && (
                                <div style={{ maxHeight: '150px', overflowY: 'auto', background: 'white', border: '1px solid #ddd', marginTop: '5px' }}>
                                    {products?.filter(p => p.name.toLowerCase().includes(productSearch.toLowerCase())).map(p => (
                                        <div
                                            key={p.id}
                                            style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid #eee' }}
                                            onClick={() => { addItem(p); setProductSearch(''); }}
                                        >
                                            {p.name} ({p.code})
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <table style={{ width: '100%', marginBottom: '15px' }}>
                            <thead>
                                <tr style={{ background: '#eee' }}>
                                    <th style={{ padding: '5px', textAlign: 'left' }}>Product</th>
                                    <th style={{ padding: '5px', width: '150px' }}>Quantity</th>
                                    <th style={{ padding: '5px' }}>Price/Cost</th>
                                    <th style={{ padding: '5px' }}>Total</th>
                                    <th style={{ padding: '5px', width: '50px' }}></th>
                                </tr>
                            </thead>
                            <tbody>
                                {formData.items.map(item => (
                                    <tr key={item.id} style={{ borderBottom: '1px solid #eee' }}>
                                        <td style={{ padding: '5px' }}>{item.name} ({item.code})</td>
                                        <td style={{ padding: '5px' }}>
                                            <input
                                                type="number"
                                                className="form-control"
                                                value={item.quantity}
                                                onChange={e => updateItemQty(item.id, e.target.value)}
                                            />
                                        </td>
                                        <td style={{ padding: '5px' }}>{formatPrice(item.cost || 0)}</td>
                                        <td style={{ padding: '5px' }}>{formatPrice((item.cost || 0) * item.quantity)}</td>
                                        <td style={{ padding: '5px' }}>
                                            <FaTrash style={{ color: 'red', cursor: 'pointer' }} onClick={() => removeItem(item.id)} />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <button type="submit" className="btn btn-primary">Submit Transfer</button>
                        <button type="button" className="btn btn-secondary" style={{ marginLeft: '10px' }} onClick={resetForm}>Cancel</button>
                    </form>
                </div>
            )}

            <div className="card">
                <div className="mb-3" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <input className="form-control" placeholder="Search reference..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                    <select className="form-control" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
                        <option value="all">All Status</option>
                        <option value="pending">Pending</option>
                        <option value="sent">Sent</option>
                        <option value="completed">Completed</option>
                    </select>
                    {hasPermission('transfers_add') && (
                        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}>
                            <label className="btn btn-secondary" style={{ cursor: 'pointer', margin: 0 }}>
                                Import CSV <input type="file" accept=".csv" hidden onChange={handleFileUpload} />
                            </label>
                        </div>
                    )}
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Reference</th>
                            <th style={{ padding: '10px' }}>From</th>
                            <th style={{ padding: '10px' }}>To</th>
                            <th style={{ padding: '10px' }}>Status</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredTransfers?.map(t => (
                            <tr key={t.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(t.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{t.reference_no}</td>
                                <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id === t.from_warehouse_id)?.name}</td>
                                <td style={{ padding: '10px' }}>{warehouses?.find(w => w.id === t.to_warehouse_id)?.name}</td>
                                <td style={{ padding: '10px' }}>
                                    <span style={{
                                        padding: '4px 8px', borderRadius: '4px', fontSize: '12px',
                                        background: t.status === 'completed' ? '#d4edda' : t.status === 'sent' ? '#fff3cd' : '#e2e3e5',
                                        color: t.status === 'completed' ? '#155724' : t.status === 'sent' ? '#856404' : '#383d41'
                                    }}>
                                        {t.status.toUpperCase()}
                                    </span>
                                </td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" onClick={() => initiatePrint(t)} title="Print"><FaPrint /></button>
                                    {t.status !== 'completed' && (
                                        <button className="btn-icon" style={{ color: 'green', marginLeft: '5px' }} onClick={async () => {
                                            if (window.confirm('Mark this transfer as COMPLETED and move stock?')) {
                                                await db.transfers.update(t.id, { status: 'completed' });
                                                await adjustStock(t.items, t.from_warehouse_id, -1);
                                                await adjustStock(t.items, t.to_warehouse_id, 1);
                                            }
                                        }} title="Mark Completed"><FaCheck /></button>
                                    )}
                                    {hasPermission('transfers_delete') && (
                                        <button className="btn-icon" style={{ color: 'red', marginLeft: '5px' }} onClick={() => handleDelete(t.id)}><FaTrash /></button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Transfers;
