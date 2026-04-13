import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';

const ProductPerformance = () => {
    const [filterDays, setFilterDays] = useState(30);

    const data = useLiveQuery(async () => {
        const products = await db.products.toArray();
        const sales = await db.sales.toArray();
        const now = new Date();
        const startDate = new Date(now.setDate(now.getDate() - filterDays));

        const productSales = {};

        // Aggregate Sales
        sales.forEach(sale => {
            if (new Date(sale.date) >= startDate) {
                sale.items.forEach(item => {
                    if (!productSales[item.id]) {
                        productSales[item.id] = 0;
                    }
                    productSales[item.id] += item.quantity;
                });
            }
        });

        // Map Results
        const results = products.map(p => ({
            ...p,
            soldQty: productSales[p.id] || 0,
            stockValue: (p.cost || 0) * (p.stock_quantity || 0)
        }));

        // Sort Fast Movers (High Sales)
        const fastMovers = [...results].sort((a, b) => b.soldQty - a.soldQty).slice(0, 20);

        // Sort Slow Movers (Low Sales but High Stock)
        // Filter: Has stock (>0) AND Low Sales (<= 5 units in period, configurable)
        // Sort by Stock Value descending (worst offenders first)
        const slowMovers = results
            .filter(p => p.stock_quantity > 0 && p.soldQty === 0)
            .sort((a, b) => b.stockValue - a.stockValue)
            .slice(0, 20);

        return { fastMovers, slowMovers };
    }, [filterDays]);

    if (!data) return <div>Loading Analysis...</div>;

    return (
        <div className="product-performance-report">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800' }}>Product Performance Analysis</h2>
                <select value={filterDays} onChange={e => setFilterDays(parseInt(e.target.value))} className="form-control" style={{ width: '150px' }}>
                    <option value="7">Last 7 Days</option>
                    <option value="30">Last 30 Days</option>
                    <option value="90">Last 3 Months</option>
                    <option value="365">Last Year</option>
                </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* Fast Movers */}
                <div className="card">
                    <h3 style={{ borderBottom: '2px solid #10b981', paddingBottom: '10px', marginTop: 0, color: '#059669' }}>🚀 Fast Moving Products</h3>
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead style={{ position: 'sticky', top: 0, background: 'var(--card-bg)' }}>
                                <tr style={{ color: 'var(--text-secondary)' }}>
                                    <th style={{ textAlign: 'left', padding: '8px' }}>Product</th>
                                    <th style={{ textAlign: 'center', padding: '8px' }}>Sold Qty</th>
                                    <th style={{ textAlign: 'right', padding: '8px' }}>Stock Left</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.fastMovers.map(p => (
                                    <tr key={p.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                        <td style={{ padding: '8px' }}>
                                            <div style={{ fontWeight: '600' }}>{p.name}</div>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{p.code}</div>
                                        </td>
                                        <td style={{ textAlign: 'center', padding: '8px', fontWeight: 'bold', color: '#10b981' }}>{p.soldQty}</td>
                                        <td style={{ textAlign: 'right', padding: '8px' }}>{p.stock_quantity}</td>
                                    </tr>
                                ))}
                                {data.fastMovers.length === 0 && <tr><td colSpan="3" style={{ padding: '20px', textAlign: 'center' }}>No sales data for this period.</td></tr>}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Slow Movers */}
                <div className="card">
                    <h3 style={{ borderBottom: '2px solid #ef4444', paddingBottom: '10px', marginTop: 0, color: '#dc2626' }}>🐌 Slow Moving (Dead Stock)</h3>
                    <p style={{ fontSize: '0.8rem', color: '#666', marginTop: '-10px', marginBottom: '10px' }}>Items with stock but NO sales in selected period. Sorted by Stock Value.</p>
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead style={{ position: 'sticky', top: 0, background: 'var(--card-bg)' }}>
                                <tr style={{ color: 'var(--text-secondary)' }}>
                                    <th style={{ textAlign: 'left', padding: '8px' }}>Product</th>
                                    <th style={{ textAlign: 'center', padding: '8px' }}>Stock Qty</th>
                                    <th style={{ textAlign: 'right', padding: '8px' }}>Value Tied Up</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.slowMovers.map(p => (
                                    <tr key={p.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                        <td style={{ padding: '8px' }}>
                                            <div style={{ fontWeight: '600' }}>{p.name}</div>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{p.code}</div>
                                        </td>
                                        <td style={{ textAlign: 'center', padding: '8px' }}>{p.stock_quantity}</td>
                                        <td style={{ textAlign: 'right', padding: '8px', fontWeight: 'bold', color: '#ef4444' }}>{p.stockValue.toFixed(2)}</td>
                                    </tr>
                                ))}
                                {data.slowMovers.length === 0 && <tr><td colSpan="3" style={{ padding: '20px', textAlign: 'center' }}>Great! No dead stock found.</td></tr>}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ProductPerformance;
