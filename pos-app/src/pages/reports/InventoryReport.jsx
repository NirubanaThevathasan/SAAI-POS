import React from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';

const InventoryReport = () => {
    const { formatPrice } = useSettings();
    const data = useLiveQuery(async () => {
        const allProducts = await db.products.toArray();
        const warehouses = await db.warehouses.toArray();
        const allStock = await db.stock.toArray();

        const totalValueRetail = allProducts.reduce((sum, p) => sum + ((p.price || 0) * (p.stock_quantity || 0)), 0);
        const totalValueCost = allProducts.reduce((sum, p) => sum + ((p.cost || 0) * (p.stock_quantity || 0)), 0);
        const potentialProfit = totalValueRetail - totalValueCost;
        const lowStockCount = allProducts.filter(p => p.stock_quantity <= (p.alert_quantity || 5)).length;

        return { allProducts, warehouses, allStock, totalValueRetail, totalValueCost, potentialProfit, lowStockCount };
    });

    if (!data) return <div>Loading...</div>;
    const { allProducts, warehouses, allStock, totalValueRetail, totalValueCost, potentialProfit, lowStockCount } = data;

    const getStockInWH = (productId, warehouseId) => {
        const entry = allStock.find(s => s.product_id == productId && s.warehouse_id == warehouseId);
        return entry ? entry.quantity : 0;
    };

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                <h3>Inventory Valuation Report</h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                <div className="card" style={{ textAlign: 'center' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Total Products</h4>
                    <div className="big-number">{allProducts.length}</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Stock Value (Retail)</h4>
                    <div className="big-number" style={{ color: 'var(--primary-color)' }}>{formatPrice(totalValueRetail)}</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Stock Value (Cost)</h4>
                    <div className="big-number" style={{ color: 'var(--accent-color)' }}>{formatPrice(totalValueCost)}</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Potential Profit</h4>
                    <div className="big-number" style={{ color: 'var(--success-color)' }}>{formatPrice(potentialProfit)}</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Low Stock Items</h4>
                    <div className="big-number" style={{ color: 'var(--warning-color)' }}>{lowStockCount}</div>
                </div>
            </div>

            <div className="card" style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '1000px' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Product</th>
                            <th style={{ padding: '10px' }}>Code</th>
                            <th style={{ padding: '10px' }}>Cost</th>
                            <th style={{ padding: '10px' }}>Price</th>
                            <th style={{ padding: '10px' }}>Stock</th>
                            {warehouses.map(w => <th key={w.id} style={{ padding: '10px', background: '#eef2ff', fontSize: '0.8rem' }}>{w.name}</th>)}
                            <th style={{ padding: '10px' }}>Inventory Value</th>
                            <th style={{ padding: '10px' }}>Potential Profit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {allProducts.map(p => {
                            const isLowStock = p.stock_quantity <= (p.alert_quantity || 5);
                            const productCostValue = (p.cost || 0) * p.stock_quantity;
                            const productRetailValue = (p.price || 0) * p.stock_quantity;
                            const productProfit = productRetailValue - productCostValue;

                            return (
                                <tr key={p.id} style={{ borderBottom: '1px solid #f9f9f9', background: isLowStock ? '#fff3cd' : 'transparent' }}>
                                    <td style={{ padding: '10px' }}>{p.name}</td>
                                    <td style={{ padding: '10px' }}>{p.code}</td>
                                    <td style={{ padding: '10px' }}>{formatPrice(p.cost || 0)}</td>
                                    <td style={{ padding: '10px' }}>{formatPrice(p.price || 0)}</td>
                                    <td style={{ padding: '10px', fontWeight: 'bold' }}>{p.stock_quantity}</td>
                                    {warehouses.map(w => (
                                        <td key={w.id} style={{ padding: '10px', background: 'rgba(238, 242, 255, 0.3)', fontSize: '0.9rem' }}>
                                            {getStockInWH(p.id, w.id)}
                                        </td>
                                    ))}
                                    <td style={{ padding: '10px' }}>
                                        <div style={{ fontSize: '0.85rem' }}>Retail: {formatPrice(productRetailValue)}</div>
                                        <div style={{ fontSize: '0.8rem', color: '#666' }}>Cost: {formatPrice(productCostValue)}</div>
                                    </td>
                                    <td style={{ padding: '10px', fontWeight: 'bold', color: 'var(--success-color)' }}>
                                        {formatPrice(productProfit)}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                <small style={{ padding: '10px', display: 'block' }}>Yellow rows indicate low stock (below alert quantity).</small>
            </div>
        </div>
    );
};

export default InventoryReport;
