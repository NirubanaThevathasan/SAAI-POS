import React from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';
import { Link } from 'react-router-dom';
import { FaShoppingCart, FaBox, FaChartLine, FaExclamationTriangle, FaArrowRight, FaClock, FaPlusCircle } from 'react-icons/fa';

const Dashboard = () => {
    const { formatPrice } = useSettings();
    const { user } = useAuth();

    // Time greeting
    const hour = new Date().getHours();
    const greeting = hour < 12 ? 'Good Morning' : hour < 18 ? 'Good Afternoon' : 'Good Evening';

    // 1. Basic Stats
    const stats = useLiveQuery(async () => {
        const products = await db.products.count();
        const sales = await db.sales.count();
        const customers = await db.customers.count();

        // Revenue (All Time)
        const allSales = await db.sales.toArray();
        const revenue = allSales.reduce((sum, s) => sum + (s.total || 0), 0);

        // Today's Revenue
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todaySales = allSales.filter(s => new Date(s.date) >= today);
        const todayRevenue = todaySales.reduce((sum, s) => sum + (s.total || 0), 0);
        const todayCount = todaySales.length;

        return { products, sales, customers, revenue, todayRevenue, todayCount };
    });

    // 2. Recent Transactions (Last 5)
    const recentSales = useLiveQuery(async () => {
        return await db.sales.orderBy('date').reverse().limit(5).toArray();
    });

    // 3. Low Stock Items (Top 5)
    const lowStockItems = useLiveQuery(async () => {
        const products = await db.products.toArray();
        return products
            .filter(p => (p.stock_quantity || 0) <= (p.alert_quantity || 5))
            .sort((a, b) => (a.stock_quantity || 0) - (b.stock_quantity || 0))
            .slice(0, 5);
    });

    return (
        <div className="dashboard">
            {/* Header Section */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '32px' }}>
                <div>
                    <h2 style={{ fontSize: '1.8rem', fontWeight: '800', margin: 0, color: 'var(--text-primary)' }}>{greeting}, {user?.username || 'Admin'}</h2>
                    <p style={{ margin: '8px 0 0', color: 'var(--text-secondary)' }}>Here's what's happening with your store today.</p>
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                    <Link to="/pos" className="btn btn-primary" style={{ padding: '10px 20px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)' }}>
                        <FaShoppingCart /> New Sale
                    </Link>
                    <Link to="/products" className="btn btn-secondary" style={{ padding: '10px 20px', borderRadius: '8px' }}>
                        <FaPlusCircle /> Add Product
                    </Link>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="dashboard-cards" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '24px', marginBottom: '32px' }}>
                <div className="card" style={{ borderLeft: '4px solid var(--primary-color)', padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: '500' }}>Today's Sales</p>
                            <h3 style={{ margin: '8px 0 0', fontSize: '1.8rem', fontWeight: '800', color: 'var(--text-primary)' }}>
                                {formatPrice(stats?.todayRevenue || 0)}
                            </h3>
                        </div>
                        <div style={{ padding: '10px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '10px', color: 'var(--primary-color)' }}>
                            <FaChartLine size={20} />
                        </div>
                    </div>
                    <span style={{ fontSize: '0.8rem', color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', alignSelf: 'flex-start', padding: '2px 8px', borderRadius: '4px', fontWeight: '600' }}>
                        {stats?.todayCount || 0} Orders Today
                    </span>
                </div>

                <div className="card" style={{ borderLeft: '4px solid var(--success-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div>
                            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Total Revenue</p>
                            <h3 style={{ margin: '8px 0 0', fontSize: '1.8rem', fontWeight: '800' }}>{formatPrice(stats?.revenue || 0)}</h3>
                        </div>
                        <div style={{ padding: '10px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '10px', color: '#10b981' }}><FaMoneyBillWaveIcon /></div>
                    </div>
                </div>

                <div className="card" style={{ borderLeft: '4px solid var(--warning-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div>
                            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Total Inventory</p>
                            <h3 style={{ margin: '8px 0 0', fontSize: '1.8rem', fontWeight: '800' }}>{stats?.products || 0}</h3>
                        </div>
                        <div style={{ padding: '10px', background: 'rgba(245, 158, 11, 0.1)', borderRadius: '10px', color: '#f59e0b' }}><FaBox size={20} /></div>
                    </div>
                </div>

                <div className="card" style={{ borderLeft: '4px solid var(--accent-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div>
                            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Customers</p>
                            <h3 style={{ margin: '8px 0 0', fontSize: '1.8rem', fontWeight: '800' }}>{stats?.customers || 0}</h3>
                        </div>
                    </div>
                </div>
            </div>

            {/* Split Section: Recent Sales & Alerts */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>

                {/* Recent Transactions */}
                <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
                    <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.01)' }}>
                        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700' }}>Recent Transactions</h3>
                        <Link to="/sales" style={{ fontSize: '0.85rem', color: 'var(--primary-color)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: '600' }}>
                            View All <FaArrowRight size={12} />
                        </Link>
                    </div>
                    <div>
                        {recentSales?.length > 0 ? (
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr style={{ background: 'var(--bg-color)', fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
                                        <th style={{ padding: '12px 24px', textAlign: 'left', fontWeight: '600' }}>ID / Date</th>
                                        <th style={{ padding: '12px 24px', textAlign: 'left', fontWeight: '600' }}>Status</th>
                                        <th style={{ padding: '12px 24px', textAlign: 'right', fontWeight: '600' }}>Amount</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {recentSales.map(sale => (
                                        <tr key={sale.id} style={{ borderBottom: '1px solid var(--border-color)', fontSize: '0.9rem' }}>
                                            <td style={{ padding: '16px 24px' }}>
                                                <div style={{ fontWeight: '600', color: 'var(--primary-color)' }}>#{sale.id}</div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <FaClock size={10} /> {new Date(sale.date).toLocaleTimeString()}
                                                </div>
                                            </td>
                                            <td style={{ padding: '16px 24px' }}>
                                                <span style={{
                                                    padding: '4px 10px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: '600', textTransform: 'uppercase',
                                                    background: sale.status === 'completed' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255, 100, 100, 0.1)',
                                                    color: sale.status === 'completed' ? '#059669' : '#dc2626'
                                                }}>
                                                    {sale.status || 'Paid'}
                                                </span>
                                            </td>
                                            <td style={{ padding: '16px 24px', textAlign: 'right', fontWeight: '700' }}>
                                                {formatPrice(sale.total)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                                No recent transactions.
                            </div>
                        )}
                    </div>
                </div>

                {/* Low Stock Alerts */}
                <div className="card" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff1f2' }}>
                        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: '#be123c', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <FaExclamationTriangle /> Low Stock
                        </h3>
                    </div>
                    <div style={{ flex: 1, padding: '10px' }}>
                        {lowStockItems?.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                {lowStockItems.map(item => (
                                    <div key={item.id} style={{
                                        padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)',
                                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        background: 'var(--bg-color)'
                                    }}>
                                        <div>
                                            <div style={{ fontWeight: '600', fontSize: '0.9rem' }}>{item.name}</div>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Code: {item.code}</div>
                                        </div>
                                        <div style={{ textAlign: 'right' }}>
                                            <div style={{ fontWeight: '800', color: '#e11d48', fontSize: '1.1rem' }}>{item.stock_quantity || 0}</div>
                                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Left</div>
                                        </div>
                                    </div>
                                ))}
                                <Link to="/reports" style={{ textAlign: 'center', marginTop: '10px', fontSize: '0.85rem', color: 'var(--primary-color)', textDecoration: 'none' }}>View Full Inventory Report</Link>
                            </div>
                        ) : (
                            <div style={{ padding: '30px', textAlign: 'center', color: '#10b981' }}>
                                <FaCheckCircle style={{ fontSize: '2rem', marginBottom: '10px', opacity: 0.5 }} />
                                <p>All Stock Levels Healthy!</p>
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
};

// Simple Icon component helper if valid
const FaMoneyBillWaveIcon = () => (
    <svg stroke="currentColor" fill="currentColor" strokeWidth="0" viewBox="0 0 512 512" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M512 80c0-26.5-21.5-48-48-48H48C21.5 32 0 53.5 0 80v352c0 26.5 21.5 48 48 48h416c26.5 0 48-21.5 48-48V80zM160 368c-26.5 0-48-21.5-48-48s21.5-48 48-48 48 21.5 48 48-21.5 48-48 48zm208 0c-26.5 0-48-21.5-48-48s21.5-48 48-48 48 21.5 48 48-21.5 48-48 48zM352 144c-26.5 0-48-21.5-48-48s21.5-48 48-48 48 21.5 48 48-21.5 48-48 48z"></path></svg>
);
import { FaCheckCircle } from 'react-icons/fa';

export default Dashboard;
