import React from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaDollarSign, FaShoppingCart, FaBox, FaExclamationTriangle, FaWallet, FaChartPie } from 'react-icons/fa';
import SalesChart from './SalesChart';
import { useSettings } from '../../context/SettingsContext';

const StatCard = ({ title, value, subtext, icon, color }) => (
    <div className="card" style={{ flex: 1, padding: '24px', borderLeft: `5px solid ${color}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', minWidth: '240px' }}>
        <div>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>{title}</h4>
            <div className="big-number" style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)' }}>{value}</div>
            {subtext && <small style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: '500' }}>{subtext}</small>}
        </div>
        <div style={{ fontSize: '36px', color: color, opacity: 0.15 }}>
            {icon}
        </div>
    </div>
);

const OverviewReport = () => {
    const { formatPrice } = useSettings();

    // Reactive Stats using useLiveQuery for instant updates
    const stats = useLiveQuery(async () => {
        const sales = await db.sales.toArray();
        const purchases = await db.purchases.toArray();
        const products = await db.products.toArray();
        const expenses = await db.expenses.toArray();

        const totalRevenue = sales.reduce((sum, s) => sum + (s.total || s.grandTotal || s.grand_total || 0), 0);
        const totalPurchases = purchases.reduce((sum, p) => sum + (p.grand_total || p.grandTotal || p.total || 0), 0);
        const totalExpenses = expenses.reduce((sum, e) => sum + (parseFloat(e.amount) || 0), 0);

        // Calculate Profit (Simple: Revenue - COGS - Expenses)
        let totalCOGS = 0;
        sales.forEach(s => {
            if (s.items) {
                s.items.forEach(item => {
                    totalCOGS += (item.cost || 0) * (item.quantity || 1);
                });
            }
        });
        const netProfit = totalRevenue - totalCOGS - totalExpenses;

        const totalInventoryValue = products.reduce((sum, p) => sum + ((p.price || 0) * (p.stock_quantity || 0)), 0);
        const lowStockCount = products.filter(p => p.stock_quantity <= (p.alert_quantity || 5)).length;

        return {
            totalSales: sales.length,
            totalRevenue,
            totalPurchases,
            totalExpenses,
            netProfit,
            totalInventoryValue,
            lowStockCount
        };
    }) || { totalSales: 0, totalRevenue: 0, totalPurchases: 0, totalExpenses: 0, netProfit: 0, totalInventoryValue: 0, lowStockCount: 0 };

    return (
        <div>
            <h2 style={{ marginBottom: '24px', fontWeight: '800', letterSpacing: '-0.025em' }}>Dashboard Overview</h2>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px', marginBottom: '32px' }}>
                <StatCard
                    title="Total Revenue"
                    value={formatPrice(stats.totalRevenue)}
                    subtext={`${stats.totalSales} Transactions`}
                    icon={<FaDollarSign />}
                    color="#2ecc71"
                />
                <StatCard
                    title="Net Profit"
                    value={formatPrice(stats.netProfit)}
                    subtext="Revenue - COGS - Expenses"
                    icon={<FaChartPie />}
                    color="#3498db"
                />
                <StatCard
                    title="Total Expenses"
                    value={formatPrice(stats.totalExpenses)}
                    icon={<FaWallet />}
                    color="#e74c3c"
                />
                <StatCard
                    title="Total Purchases"
                    value={formatPrice(stats.totalPurchases)}
                    icon={<FaShoppingCart />}
                    color="#f39c12"
                />
                <StatCard
                    title="Inventory Value"
                    value={formatPrice(stats.totalInventoryValue)}
                    subtext="Retail Value"
                    icon={<FaBox />}
                    color="var(--primary-color)"
                />
                <StatCard
                    title="Low Stock Alerts"
                    value={stats.lowStockCount}
                    subtext="Items to Reorder"
                    icon={<FaExclamationTriangle />}
                    color="#95a5a6"
                />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                <div className="card">
                    <h3 style={{ marginBottom: '20px' }}>Sales Trend</h3>
                    <SalesChart />
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '20px' }}>Quick Actions</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                        <p style={{ color: '#666', gridColumn: 'span 2' }}>Recent activity and key shortcuts will appear here.</p>
                        <div style={{ padding: '15px', background: '#f8f9fa', borderRadius: '12px', textAlign: 'center' }}>
                            <div style={{ fontWeight: 'bold' }}>{stats.totalSales}</div>
                            <small>Total Sales</small>
                        </div>
                        <div style={{ padding: '15px', background: '#f8f9fa', borderRadius: '12px', textAlign: 'center' }}>
                            <div style={{ fontWeight: 'bold' }}>{stats.lowStockCount}</div>
                            <small>Low Stock</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default OverviewReport;
