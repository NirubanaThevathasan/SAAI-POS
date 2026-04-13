import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';
import { FaChartLine, FaArrowUp, FaArrowDown, FaMinusCircle } from 'react-icons/fa';

const ProfitReport = () => {
    const { formatPrice } = useSettings();
    const [dateRange, setDateRange] = useState(() => {
        const now = new Date();
        const firstDay = new Date(now.getFullYear(), now.getMonth(), 1).toISOString().split('T')[0];
        const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0).toISOString().split('T')[0];
        return { start: firstDay, end: lastDay };
    });

    const stats = useLiveQuery(async () => {
        if (!dateRange.start || !dateRange.end) return null;

        const start = new Date(dateRange.start);
        const end = new Date(dateRange.end);
        end.setHours(23, 59, 59);

        // Efficiency: Use date filter in query if possible, but date is stored as Object/String in some cases.
        // For now, fetching all and filtering in JS to maintain compatibility with existing data formats.
        const sales = await db.sales.toArray();
        const expenses = await db.expenses.toArray();
        const purchases = await db.purchases.toArray();
        const returns = await db.returns.toArray();

        const filteredSales = sales.filter(s => {
            const d = new Date(s.date);
            return d >= start && d <= end;
        });

        const filteredExpenses = expenses.filter(e => {
            const d = new Date(e.date);
            return d >= start && d <= end;
        });

        const filteredPurchases = purchases.filter(p => {
            const d = new Date(p.date);
            return d >= start && d <= end;
        });

        const filteredReturns = returns.filter(r => {
            const d = new Date(r.date);
            return d >= start && d <= end;
        });

        // 1. Revenue & COGS from Sales
        let grossRevenue = 0;
        let saleCOGS = 0;

        filteredSales.forEach(sale => {
            grossRevenue += (sale.total || sale.grandTotal || sale.grand_total || 0);
            if (sale.items) {
                sale.items.forEach(item => {
                    saleCOGS += (item.cost || 0) * (item.quantity || 1);
                });
            }
        });

        // 2. Adjustments for Sale Returns
        let saleReturnsTotal = 0;
        let saleReturnsCOGS = 0;

        filteredReturns.filter(r => r.type === 'sale').forEach(ret => {
            saleReturnsTotal += (ret.totalRefund || ret.total || 0);
            if (ret.items) {
                ret.items.forEach(item => {
                    saleReturnsCOGS += (item.cost || 0) * (item.quantity || 1);
                });
            }
        });

        // 3. Actual Expenditures
        const totalPurchasesExpenditure = filteredPurchases.reduce((sum, p) => sum + (p.grand_total || p.grandTotal || p.total || 0), 0);
        const totalExpenses = filteredExpenses.reduce((sum, e) => sum + (parseFloat(e.amount) || 0), 0);

        // Financial Metrics
        const netRevenue = grossRevenue - saleReturnsTotal;
        const netCOGS = saleCOGS - saleReturnsCOGS;
        const grossProfit = netRevenue - netCOGS;
        const netProfit = grossProfit - totalExpenses;

        return {
            revenue: netRevenue,
            cogs: netCOGS,
            grossProfit,
            expenses: totalExpenses,
            netProfit,
            purchases: totalPurchasesExpenditure,
            saleReturns: saleReturnsTotal,
            count: filteredSales.length
        };
    }, [dateRange]);

    if (!stats) return <div className="p-4 text-center">Loading profit data...</div>;

    const profitMargin = stats.revenue > 0 ? (stats.netProfit / stats.revenue) * 100 : 0;

    return (
        <div style={{ animation: 'fadeIn 0.4s ease-out' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                <div>
                    <h2 style={{ fontSize: '1.75rem', fontWeight: '800', letterSpacing: '-0.025em', marginBottom: '4px' }}>Profit & Loss</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Comprehensive financial performance tracking</p>
                </div>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', background: '#fff', padding: '8px 16px', borderRadius: '16px', boxShadow: 'var(--shadow-premium)' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-muted)' }}>FROM</span>
                        <input type="date" className="form-control" style={{ border: 'none', padding: '4px', fontSize: '0.9rem', width: '130px' }} value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
                    </div>
                    <div style={{ width: '1px', height: '20px', background: '#eee' }}></div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-muted)' }}>TO</span>
                        <input type="date" className="form-control" style={{ border: 'none', padding: '4px', fontSize: '0.9rem', width: '130px' }} value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
                    </div>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '32px' }}>
                {/* Main Profit Card */}
                <div className="card" style={{
                    background: 'linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%)',
                    color: '#fff',
                    padding: '32px',
                    position: 'relative',
                    overflow: 'hidden'
                }}>
                    <FaChartLine style={{ position: 'absolute', right: '-20px', bottom: '-20px', fontSize: '120px', opacity: 0.1 }} />
                    <h3 style={{ fontSize: '1rem', fontWeight: '500', opacity: 0.8, marginBottom: '8px' }}>Net Profit</h3>
                    <div style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '16px' }}>{formatPrice(stats.netProfit)}</div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <span style={{
                            background: 'rgba(255,255,255,0.2)',
                            padding: '4px 12px',
                            borderRadius: '20px',
                            fontSize: '0.85rem',
                            fontWeight: '600',
                            backdropFilter: 'blur(4px)'
                        }}>
                            {profitMargin.toFixed(1)}% Margin
                        </span>
                        <span style={{ opacity: 0.7, fontSize: '0.85rem' }}>on {stats.count} sales</span>
                    </div>
                </div>

                {/* Breakdown Summary */}
                <div className="card" style={{ padding: '0', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '24px', borderBottom: '1px solid #f1f5f9' }}>
                        <h4 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-primary)' }}>Breakdown</h4>
                    </div>
                    <div style={{ padding: '12px 24px' }}>
                        {[
                            { label: 'Net Revenue', value: stats.revenue, icon: <FaArrowUp />, color: 'var(--success-color)' },
                            { label: 'Cost of Goods Sold', value: stats.cogs, icon: <FaArrowDown />, color: 'var(--accent-color)' },
                            { label: 'Expenses', value: stats.expenses, icon: <FaMinusCircle />, color: 'var(--warning-color)' },
                            { label: 'Purchase Expenditure', value: stats.purchases, icon: <FaArrowDown />, color: '#64748b', isSecondary: true },
                        ].map((item, i) => (
                            <div key={i} style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                padding: '12px 0',
                                borderTop: item.isSecondary ? '1px dashed #eee' : 'none',
                                marginTop: item.isSecondary ? '8px' : '0',
                                opacity: item.isSecondary ? 0.7 : 1
                            }}>
                                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: `${item.color}15`, color: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem' }}>
                                        {item.icon}
                                    </div>
                                    <span style={{ fontSize: '0.95rem', fontWeight: '500', color: 'var(--text-secondary)' }}>
                                        {item.label}
                                        {item.isSecondary && <small style={{ display: 'block', fontSize: '0.7rem' }}>Total cost of stock bought</small>}
                                    </span>
                                </div>
                                <span style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                                    {i > 0 && !item.isSecondary && '-'} {formatPrice(item.value)}
                                </span>
                            </div>
                        ))}
                    </div>
                    <div style={{ marginTop: 'auto', padding: '20px 24px', background: '#f8fafc', borderBottomLeftRadius: '24px', borderBottomRightRadius: '24px', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: '600', color: 'var(--text-secondary)' }}>Gross Profit</span>
                        <span style={{ fontWeight: '800', color: 'var(--success-color)' }}>{formatPrice(stats.grossProfit)}</span>
                    </div>
                </div>
            </div>

            <div className="card" style={{ padding: '32px' }}>
                <h4 style={{ marginBottom: '20px', fontWeight: '700' }}>Historical Analysis</h4>
                <p style={{ color: 'var(--text-muted)' }}>Detailed monthly trends and category breakdowns are being aggregated and will appear here shortly.</p>
                <div style={{ height: '200px', background: '#f1f5f9', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed #e2e8f0' }}>
                    <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Financial Charts Loading...</span>
                </div>
            </div>

            <p style={{ textAlign: 'center', marginTop: '32px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                * Profit calculation accounts for verified sale records and documented business expenses within the selected period.
            </p>
        </div>
    );
};

export default ProfitReport;
