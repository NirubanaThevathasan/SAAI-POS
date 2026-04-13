import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { FaCalendarDay, FaCalendarAlt, FaCalendarCheck } from 'react-icons/fa';

const PeriodicSalesReport = () => {
    const { formatPrice } = useSettings();
    const [periodType, setPeriodType] = useState('daily'); // 'daily' | 'monthly' | 'yearly'

    const data = useLiveQuery(async () => {
        const sales = await db.sales.toArray();
        const stats = {};

        sales.forEach(sale => {
            const date = new Date(sale.date);
            let key;
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');

            if (periodType === 'daily') {
                key = `${year}-${month}-${day}`;
            } else if (periodType === 'monthly') {
                key = `${year}-${month}`;
            } else {
                key = `${year}`;
            }

            const total = (sale.total || sale.grandTotal || sale.grand_total || 0);
            if (!stats[key]) {
                stats[key] = { label: key, total: 0, count: 0 };
            }
            stats[key].total += total;
            stats[key].count += 1;
        });

        // Convert to array and sort
        const result = Object.values(stats).sort((a, b) => a.label.localeCompare(b.label));

        // Take last N for readability
        const limit = periodType === 'daily' ? 14 : (periodType === 'monthly' ? 12 : 5);
        return result.slice(-limit);
    }, [periodType]);

    if (!data) return <div className="p-4 text-center">Loading periodic data...</div>;

    return (
        <div style={{ animation: 'fadeIn 0.4s ease-out' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '800', letterSpacing: '-0.025em' }}>Periodic Sales Summary</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Comparative analysis across different timeframes</p>
                </div>

                <div style={{ background: '#fff', padding: '4px', borderRadius: '12px', display: 'flex', gap: '4px', boxShadow: 'var(--shadow-premium)' }}>
                    <button
                        onClick={() => setPeriodType('daily')}
                        style={{
                            padding: '8px 16px',
                            border: 'none',
                            borderRadius: '8px',
                            background: periodType === 'daily' ? 'var(--primary-color)' : 'transparent',
                            color: periodType === 'daily' ? '#fff' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '8px',
                            transition: 'all 0.2s'
                        }}
                    >
                        <FaCalendarDay size={14} /> Daily
                    </button>
                    <button
                        onClick={() => setPeriodType('monthly')}
                        style={{
                            padding: '8px 16px',
                            border: 'none',
                            borderRadius: '8px',
                            background: periodType === 'monthly' ? 'var(--primary-color)' : 'transparent',
                            color: periodType === 'monthly' ? '#fff' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '8px',
                            transition: 'all 0.2s'
                        }}
                    >
                        <FaCalendarAlt size={14} /> Monthly
                    </button>
                    <button
                        onClick={() => setPeriodType('yearly')}
                        style={{
                            padding: '8px 16px',
                            border: 'none',
                            borderRadius: '8px',
                            background: periodType === 'yearly' ? 'var(--primary-color)' : 'transparent',
                            color: periodType === 'yearly' ? '#fff' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '8px',
                            transition: 'all 0.2s'
                        }}
                    >
                        <FaCalendarCheck size={14} /> Yearly
                    </button>
                </div>
            </div>

            <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
                <h4 style={{ marginBottom: '20px', fontWeight: '700' }}>Sales Performance Chart</h4>
                <div style={{ height: '350px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                            <XAxis
                                dataKey="label"
                                axisLine={false}
                                tickLine={false}
                                tick={{ fill: '#64748b', fontSize: 12 }}
                            />
                            <YAxis
                                axisLine={false}
                                tickLine={false}
                                tick={{ fill: '#64748b', fontSize: 12 }}
                                tickFormatter={(value) => formatPrice(value).replace('Rs. ', '')}
                            />
                            <Tooltip
                                cursor={{ fill: '#f8fafc' }}
                                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: 'var(--shadow-premium)' }}
                                formatter={(value) => [formatPrice(value), 'Revenue']}
                            />
                            <Legend verticalAlign="top" align="right" />
                            <Bar
                                dataKey="total"
                                name="Revenue"
                                fill="var(--primary-color)"
                                radius={[6, 6, 0, 0]}
                                barSize={periodType === 'daily' ? 30 : (periodType === 'monthly' ? 50 : 80)}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="card" style={{ padding: '0' }}>
                <div style={{ padding: '24px', borderBottom: '1px solid #f1f5f9' }}>
                    <h4 style={{ fontWeight: '700' }}>Breakdown Table</h4>
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                            <th style={{ padding: '16px 24px', color: '#64748b', fontWeight: '600', fontSize: '0.85rem', textTransform: 'uppercase' }}>Period</th>
                            <th style={{ padding: '16px 24px', color: '#64748b', fontWeight: '600', fontSize: '0.85rem', textTransform: 'uppercase' }}>Transaction Count</th>
                            <th style={{ padding: '16px 24px', textAlign: 'right', color: '#64748b', fontWeight: '600', fontSize: '0.85rem', textTransform: 'uppercase' }}>Total Revenue</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.slice().reverse().map((item, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                <td style={{ padding: '16px 24px', fontWeight: '600', color: 'var(--text-primary)' }}>{item.label}</td>
                                <td style={{ padding: '16px 24px', color: 'var(--text-secondary)' }}>{item.count} Sales</td>
                                <td style={{ padding: '16px 24px', textAlign: 'right', fontWeight: '700', color: 'var(--primary-color)' }}>{formatPrice(item.total)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PeriodicSalesReport;
