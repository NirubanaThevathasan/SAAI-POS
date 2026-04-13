import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { FaTrash, FaEye, FaPrint, FaSearch, FaEdit, FaTruck } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { useSettings } from '../../context/SettingsContext';
import { useAuth } from '../../context/AuthContext';

const SalesList = () => {
    const { settings, formatPrice } = useSettings();
    const { hasPermission } = useAuth();
    const navigate = useNavigate();
    const sales = useLiveQuery(() => db.sales.orderBy('date').reverse().toArray());
    const customers = useLiveQuery(() => db.customers.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());

    const [searchTerm, setSearchTerm] = useState('');
    const [filterCustomer, setFilterCustomer] = useState('');
    const [filterWarehouse, setFilterWarehouse] = useState('');
    const [dateRange, setDateRange] = useState({ start: '', end: '' });

    const [selectedSale, setSelectedSale] = useState(null); // For details modal

    const [filterSource, setFilterSource] = useState('all'); // 'all', 'pos', 'csv_import'

    const filteredSales = sales?.filter(sale => {
        const saleDate = new Date(sale.date);
        const start = dateRange.start ? new Date(dateRange.start) : null;
        const end = dateRange.end ? new Date(dateRange.end) : null;
        if (end) end.setHours(23, 59, 59); // End of day

        const matchesSearch = searchTerm === '' ||
            (sale.id.toString().includes(searchTerm) ||
                (sale.customer_id && customers?.find(c => c.id === sale.customer_id)?.name.toLowerCase().includes(searchTerm.toLowerCase())));

        const matchesCustomer = filterCustomer ? sale.customer_id === parseInt(filterCustomer) : true;
        const matchesWarehouse = filterWarehouse ? sale.warehouse_id === parseInt(filterWarehouse) : true;

        const matchesDate = (!start || saleDate >= start) && (!end || saleDate <= end);

        const matchesSource = filterSource === 'all' ? true : (sale.source || 'pos') === filterSource;

        return matchesSearch && matchesCustomer && matchesWarehouse && matchesDate && matchesSource;
    });

    // Revert stock helper
    const revertStock = async (items, warehouseId) => {
        const whId = parseInt(warehouseId || 1); // Default to WH 1 if not set
        for (const item of items) {
            // 1. Revert Warehouse Stock
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

            // 2. Revert Global Stock
            const product = await db.products.get(item.id);
            if (product) {
                await db.products.update(item.id, {
                    stock_quantity: (product.stock_quantity || 0) + item.quantity
                });
            }
        }
    };

    const handleDelete = async (sale) => {
        if (window.confirm('Delete this sale? Stock WILL be reverted.')) {
            await revertStock(sale.items, sale.warehouse_id);
            await db.sales.delete(sale.id);
        }
    }

    const handleEdit = async (sale) => {
        if (!window.confirm('Edit this sale? It will be moved to "Suspended Sales" in POS for modification. Stock will be reverted.')) return;

        await revertStock(sale.items, sale.warehouse_id);

        const customerName = sale.customer_id ? customers?.find(c => c.id === sale.customer_id)?.name : 'Walk-in';

        await db.suspended_sales.add({
            date: new Date(),
            items: sale.items,
            customer_id: sale.customer_id,
            customer_name: customerName,
            total: sale.total,
            warehouse_id: sale.warehouse_id // Maintain warehouse context
        });

        await db.sales.delete(sale.id);
        navigate('/pos');
    }

    const handleMarkDelivery = async (sale) => {
        const existing = await db.deliveries.where('sale_id').equals(sale.id).first();
        if (existing) return alert('Delivery record already exists for this sale.');

        const customer = sale.customer_id ? customers?.find(c => c.id === sale.customer_id) : null;

        await db.deliveries.add({
            sale_id: sale.id,
            date: new Date(),
            status: 'packing',
            customer_id: sale.customer_id,
            address: customer?.address || '',
            note: '',
            reference_no: ''
        });
        alert('Delivery record created!');
        navigate('/sales/deliveries');
    }

    // CSV Import
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
                // Expected format: Date,CustomerName,Total,Status,PaymentMethod
                const [dateStr, customerName, totalStr, status, method] = row.split(',');

                // Attempt to find customer by name, or use default/walk-in
                const customer = await db.customers.where('name').equalsIgnoreCase(customerName?.trim() || '').first();

                await db.sales.add({
                    date: new Date(dateStr || new Date()),
                    total: parseFloat(totalStr) || 0,
                    status: status?.trim() || 'completed',
                    paymentMethod: method?.trim() || 'Cash',
                    source: 'csv_import',
                    customer_id: customer?.id || null, // null = walk-in
                    items: [] // CSV simple import might not have line items
                });
                count++;
            }
            alert(`Imported ${count} sales successfully!`);
            e.target.value = ''; // Reset input
        };
        reader.readAsText(file);
    };

    // Invoice Print Helper (Simplified)
    const handlePrint = () => {
        window.print();
    }

    return (
        <div className="card">
            {/* Filters */}
            <div className="mb-3 p-3 bg-light rounded" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div style={{ flex: 1, minWidth: '200px' }}>
                    <label>Search</label>
                    <input className="form-control" placeholder="Search ID or Customer..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                </div>
                <div>
                    <label>Customer</label>
                    <select className="form-control" value={filterCustomer} onChange={e => setFilterCustomer(e.target.value)}>
                        <option value="">All Customers</option>
                        {customers?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                    </select>
                </div>
                <div>
                    <label>Warehouse</label>
                    <select className="form-control" value={filterWarehouse} onChange={e => setFilterWarehouse(e.target.value)}>
                        <option value="">All Warehouses</option>
                        {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
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

                <div>
                    <label>Source</label>
                    <select className="form-control" value={filterSource} onChange={e => setFilterSource(e.target.value)}>
                        <option value="all">All Sources</option>
                        <option value="pos">POS Sales</option>
                        <option value="csv_import">Imported (CSV)</option>
                    </select>
                </div>

                {hasPermission('sales_add') && (
                    <div style={{ marginLeft: 'auto', display: 'flex', flexDirection: 'column' }}>
                        <label>Import Sales (CSV)</label>
                        <input type="file" accept=".csv" onChange={handleFileUpload} className="form-control" style={{ width: '250px' }} />
                        <small style={{ fontSize: '10px', color: '#666' }}>Format: Date,Customer,Total,Status,Method</small>
                    </div>
                )}
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                    <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                        <th style={{ padding: '10px' }}>Date</th>
                        <th style={{ padding: '10px' }}>Customer</th>
                        <th style={{ padding: '10px' }}>Warehouse</th>
                        <th style={{ padding: '10px' }}>Items</th>
                        <th style={{ padding: '10px' }}>Total</th>
                        <th style={{ padding: '10px' }}>Status</th>
                        <th style={{ padding: '10px' }}>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {filteredSales?.map(sale => (
                        <tr key={sale.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                            <td style={{ padding: '10px' }}>{new Date(sale.date).toLocaleString()}</td>
                            <td style={{ padding: '10px' }}>
                                {sale.customer_id ? customers?.find(c => c.id === sale.customer_id)?.name : 'Walk-in'}
                            </td>
                            <td style={{ padding: '10px' }}>
                                {sale.warehouse_id ? warehouses?.find(w => w.id === sale.warehouse_id)?.name : 'Default'}
                            </td>
                            <td style={{ padding: '10px' }}>{sale.items?.length || 0} items</td>
                            <td style={{ padding: '10px' }}>{formatPrice(sale.total || 0)}</td>
                            <td style={{ padding: '10px' }}>
                                <span style={{ padding: '4px 8px', borderRadius: '4px', background: '#e1f5fe', color: '#0288d1' }}>
                                    {sale.status}
                                </span>
                            </td>
                            <td style={{ padding: '10px', display: 'flex', gap: '8px' }}>
                                <button className="btn-icon" onClick={() => setSelectedSale(sale)} title="View Details"><FaEye /></button>
                                {hasPermission('sales_edit') && (
                                    <button className="btn-icon" style={{ color: 'var(--primary-color)' }} onClick={() => handleMarkDelivery(sale)} title="Mark for Delivery"><FaTruck /></button>
                                )}
                                {hasPermission('sales_edit') && (
                                    <button className="btn-icon" style={{ color: 'orange' }} onClick={() => handleEdit(sale)} title="Edit"><FaEdit /></button>
                                )}
                                {hasPermission('sales_delete') && (
                                    <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(sale)} title="Delete"><FaTrash /></button>
                                )}
                            </td>
                        </tr>
                    ))}
                    {filteredSales?.length === 0 && (
                        <tr>
                            <td colSpan="7" style={{ padding: '20px', textAlign: 'center' }}>No sales found matching criteria.</td>
                        </tr>
                    )}
                </tbody>
            </table>

            {/* Details Modal */}
            {selectedSale && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
                }}>
                    <div className="modal-content card" style={{ padding: '20px', width: '600px', maxHeight: '90vh', overflowY: 'auto' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                            <h3>Sale #{selectedSale.id} Details</h3>
                            <button className="btn btn-secondary" onClick={() => setSelectedSale(null)}>Close</button>
                        </div>

                        <div id="invoice-print-area">
                            <div style={{ textAlign: 'center', marginBottom: '20px', borderBottom: '1px dashed #ccc', paddingBottom: '10px' }}>
                                <h1 style={{ margin: '0 0 5px 0', fontSize: '1.5rem', textTransform: 'uppercase' }}>{settings.site_name || 'INVOICE'}</h1>
                                {settings.pos_receipt_header && <p style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#666' }}>{settings.pos_receipt_header}</p>}
                                <div style={{ borderTop: '1px solid #eee', paddingTop: '10px', marginTop: '10px' }}>
                                    <p style={{ margin: '2px 0' }}>Date: {new Date(selectedSale.date).toLocaleString()}</p>
                                    <p style={{ margin: '2px 0' }}>Customer: {selectedSale.customer_id ? customers?.find(c => c.id === selectedSale.customer_id)?.name : 'Walk-in'}</p>
                                </div>
                            </div>

                            <table style={{ width: '100%', marginBottom: '20px' }}>
                                <thead>
                                    <tr style={{ borderBottom: '1px solid #000' }}><th>Item</th><th>Qty</th><th>Price</th><th>Total</th></tr>
                                </thead>
                                <tbody>
                                    {selectedSale.items?.map((item, i) => (
                                        <tr key={i}>
                                            <td>{item.name}</td>
                                            <td>{item.quantity}</td>
                                            <td>{formatPrice(item.price)}</td>
                                            <td>{formatPrice(item.price * item.quantity)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>

                            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px', borderTop: '1px dashed #ccc', paddingTop: '10px' }}>
                                <div style={{ textAlign: 'right' }}>
                                    <p>Subtotal: {formatPrice(selectedSale.subtotal || 0)}</p>
                                    <p>Tax: {formatPrice(selectedSale.tax || 0)}</p>
                                    <p>Discount: -{formatPrice(selectedSale.discount || 0)}</p>
                                    <h3 style={{ margin: '5px 0' }}>Total: {formatPrice(selectedSale.total)}</h3>
                                    <p>Paid: {formatPrice(selectedSale.received || selectedSale.total)}</p>
                                </div>
                            </div>

                            {settings.pos_receipt_footer && (
                                <div style={{ textAlign: 'center', marginTop: '20px', borderTop: '1px dashed #ccc', paddingTop: '10px', fontSize: '0.85rem', color: '#666' }}>
                                    <p>{settings.pos_receipt_footer}</p>
                                </div>
                            )}
                        </div>

                        <div className="no-print" style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
                            <button className="btn btn-primary" onClick={handlePrint}><FaPrint /> Print Invoice</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Print Styles */}
            <style>{`
                @media print {
                    body * { visibility: hidden; }
                    .modal-overlay, .modal-content, #invoice-print-area, #invoice-print-area * { visibility: visible; }
                    .modal-overlay { position: absolute; left: 0; top: 0; background: white; }
                    .modal-content { box-shadow: none; border: none; width: 100%; }
                    .no-print { display: none; }
                }
            `}</style>
        </div>
    );
};

export default SalesList;
