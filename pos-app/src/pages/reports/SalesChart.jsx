import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useSettings } from '../../context/SettingsContext';

const SalesChart = ({ data: salesProp }) => {
    const { formatPrice } = useSettings();

    // Aggregate sales by date if they exist
    const grouped = {};
    if (salesProp) {
        salesProp.forEach(sale => {
            const date = new Date(sale.date).toLocaleDateString();
            grouped[date] = (grouped[date] || 0) + (sale.grandTotal || sale.total || 0);
        });
    }

    const chartData = Object.keys(grouped).map(date => ({
        date,
        sales: grouped[date]
    })).sort((a, b) => new Date(a.date) - new Date(b.date));

    return (
        <div style={{ height: '350px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis
                        dataKey="date"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        dy={10}
                    />
                    <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        tickFormatter={(value) => formatPrice(value)}
                    />
                    <Tooltip
                        contentStyle={{
                            borderRadius: '12px',
                            border: 'none',
                            boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
                            fontSize: '12px'
                        }}
                        formatter={(value) => [formatPrice(value), 'Revenue']}
                    />
                    <Bar
                        dataKey="sales"
                        fill="var(--primary-color)"
                        radius={[6, 6, 0, 0]}
                        barSize={40}
                        name="Daily Revenue"
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SalesChart;
