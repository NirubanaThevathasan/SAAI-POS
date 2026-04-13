import React, { useState } from 'react';
import { db } from '../../db';
import SalesChart from './SalesChart';
import { useSettings } from '../../context/SettingsContext';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaChartPie, FaList, FaFileDownload } from 'react-icons/fa';

const SalesReport = () => {
    const { formatPrice } = useSettings();
    const [dateRange, setDateRange] = useState({
        start: new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0],
        end: new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).toISOString().split('T')[0]
    });
    const [viewMode, setViewMode] = useState('summary'); // summary, product, category

    const data = useLiveQuery(async () => {
        if (!dateRange.start || !dateRange.end) return null;

        const start = new Date(dateRange.start);
        const end = new Date(dateRange.end);
        end.setHours(23, 59, 59);

        const allSales = await db.sales.toArray();
        const customers = await db.customers.toArray();
        const products = await db.products.toArray();
        const categories = await db.categories.toArray();

        const customerMap = new Map(customers.map(c => [c.id, c.name]));
        const categoryMap = new Map(categories.map(c => [c.id, c.name]));
        const productCategoryMap = new Map(products.map(p => [p.id, p.category_id]));

        const relevantSales = allSales.filter(s => {
            const d = new Date(s.date);
            return d >= start && d <= end;
        });

        const totalRevenue = relevantSales.reduce((sum, s) => sum + (s.total || 0), 0);

        // Aggregation
        const productStats = {};
        const categoryStats = {};

        relevantSales.forEach(sale => {
            if (sale.items) {
                sale.items.forEach(item => {
                    const revenue = (item.price * item.quantity) || 0;

                    // Product Stats
                    const pName = item.name || 'Unknown';
                    if (!productStats[pName]) productStats[pName] = { qty: 0, revenue: 0 };
                    productStats[pName].qty += item.quantity;
                    productStats[pName].revenue += revenue;

                    // Category Stats
                    // Need to find product category. item.id -> product -> category_id -> category name
                    let catName = 'Uncategorized';
                    if (item.category) {
                        catName = item.category; // If saved in sale item
                    } else if (item.id) {
                        const catId = productCategoryMap.get(item.id);
                        if (catId) catName = categoryMap.get(catId) || 'Unknown Category';
                    }

                    if (!categoryStats[catName]) categoryStats[catName] = { qty: 0, revenue: 0 };
                    categoryStats[catName].qty += item.quantity;
                    categoryStats[catName].revenue += revenue;
                });
            }
        });

        // Convert to Arrays & Sort
        const productList = Object.entries(productStats)
            .map(([name, d]) => ({ name, ...d }))
            .sort((a, b) => b.revenue - a.revenue);

        const categoryList = Object.entries(categoryStats)
            .map(([name, d]) => ({ name, ...d }))
            .sort((a, b) => b.revenue - a.revenue);

        return { relevantSales, totalRevenue, productList, categoryList, customerMap };
    }, [dateRange]);

    if (!data) return <div>Loading Sales Report...</div>;
    const { relevantSales, totalRevenue, productList, categoryList, customerMap } = data;

    const maxCategoryRevenue = Math.max(...categoryList.map(c => c.revenue), 1);

    return (
        <div className="report-container">
            {/* Header / Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '15px' }}>
                <div>
                    <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800' }}>Sales Report</h2>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                        <button onClick={() => setViewMode('summary')} className={`btn ${viewMode === 'summary' ? 'btn-primary' : 'btn-secondary'}`} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>Overview</button>
                        <button onClick={() => setViewMode('product')} className={`btn ${viewMode === 'product' ? 'btn-primary' : 'btn-secondary'}`} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>Product-wise</button>
                        <button onClick={() => setViewMode('category')} className={`btn ${viewMode === 'category' ? 'btn-primary' : 'btn-secondary'}`} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>Category-wise</button>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', background: 'var(--card-bg)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <input type="date" className="form-control" style={{ width: 'auto' }} value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
                    <span style={{ color: 'var(--text-secondary)' }}>to</span>
                    <input type="date" className="form-control" style={{ width: 'auto' }} value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
                </div>
            </div>

            {/* Content Based on View Mode */}

            {viewMode === 'summary' && (
                <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '24px' }}>
                        <div className="card">
                            <h4 style={{ margin: '0 0 15px 0', color: 'var(--text-secondary)' }}>Sales Trend</h4>
                            <SalesChart data={relevantSales} />
                        </div>
                        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                            <div style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>Total Revenue</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '800', color: 'var(--success-color)' }}>{formatPrice(totalRevenue)}</div>
                            <div style={{ marginTop: '10px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{relevantSales.length} Transactions</div>
                        </div>
                    </div>

                    <div className="card">
                        <h4 style={{ marginBottom: '16px' }}>Sales Log</h4>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left', background: 'var(--bg-color)' }}>
                                        <th style={{ padding: '12px' }}>Date</th>
                                        <th style={{ padding: '12px' }}>Ref</th>
                                        <th style={{ padding: '12px' }}>Customer</th>
                                        <th style={{ padding: '12px' }}>Status</th>
                                        <th style={{ padding: '12px' }}>Method</th>
                                        <th style={{ padding: '12px', textAlign: 'right' }}>Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {relevantSales.map(sale => (
                                        <tr key={sale.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <td style={{ padding: '12px' }}>{new Date(sale.date).toLocaleDateString()}</td>
                                            <td style={{ padding: '12px' }}>#{sale.id}</td>
                                            <td style={{ padding: '12px' }}>{sale.customer_name || (sale.customer_id && customerMap.get(sale.customer_id)) || 'Walk-in'}</td>
                                            <td style={{ padding: '12px' }}>
                                                <span style={{ padding: '4px 8px', borderRadius: '4px', background: '#d1fae5', color: '#065f46', fontSize: '0.8rem', fontWeight: '600' }}>{sale.status}</span>
                                            </td>
                                            <td style={{ padding: '12px' }}>{sale.paymentMethod || 'Cash'}</td>
                                            <td style={{ padding: '12px', textAlign: 'right', fontWeight: '600' }}>{formatPrice(sale.total)}</td>
                                        </tr>
                                    ))}
                                    {relevantSales.length === 0 && <tr><td colSpan="6" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>No sales found in this period.</td></tr>}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}

            {viewMode === 'product' && (
                <div className="card">
                    <h3 style={{ margin: '0 0 20px 0' }}>📦 Product-wise Sales Performance</h3>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ background: 'var(--bg-color)', borderBottom: '2px solid var(--border-color)', textAlign: 'left' }}>
                                <th style={{ padding: '12px' }}>Product Name</th>
                                <th style={{ padding: '12px', textAlign: 'center' }}>Quantity Sold</th>
                                <th style={{ padding: '12px', textAlign: 'right' }}>Total Revenue</th>
                                <th style={{ padding: '12px', textAlign: 'right' }}>% of Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            {productList.map((p, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                    <td style={{ padding: '12px', fontWeight: '500' }}>{p.name}</td>
                                    <td style={{ padding: '12px', textAlign: 'center' }}>{p.qty}</td>
                                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: '600' }}>{formatPrice(p.revenue)}</td>
                                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                                        {((p.revenue / totalRevenue) * 100).toFixed(1)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {viewMode === 'category' && (
                <div className="card">
                    <h3 style={{ margin: '0 0 20px 0' }}>📂 Category-wise Sales Performance</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
                        {/* Visual Chart Bars */}
                        <div>
                            {categoryList.map((c, i) => (
                                <div key={i} style={{ marginBottom: '16px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '0.9rem', fontWeight: '600' }}>
                                        <span>{c.name}</span>
                                        <span>{formatPrice(c.revenue)}</span>
                                    </div>
                                    <div style={{ width: '100%', height: '24px', background: 'var(--bg-color)', borderRadius: '12px', overflow: 'hidden' }}>
                                        <div style={{
                                            width: `${(c.revenue / maxCategoryRevenue) * 100}%`,
                                            height: '100%',
                                            background: `hsl(${i * 45}, 70%, 50%)`, // Auto-generate colors
                                            borderRadius: '12px',
                                            transition: 'width 0.5s ease-out'
                                        }}></div>
                                    </div>
                                    <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                                        {c.qty} items sold
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SalesReport;
