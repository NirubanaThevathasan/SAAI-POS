import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useSettings } from '../../context/SettingsContext';

const ExpenseReport = () => {
    const { formatPrice } = useSettings();
    const expenses = useLiveQuery(() => db.expenses.toArray());
    const categories = useLiveQuery(() => db.expense_categories.toArray());

    const [dateRange, setDateRange] = useState({ start: '', end: '' });

    // Filter by date
    const filteredExpenses = expenses?.filter(e => {
        const eDate = new Date(e.date);
        const start = dateRange.start ? new Date(dateRange.start) : null;
        const end = dateRange.end ? new Date(dateRange.end) : null;
        if (end) end.setHours(23, 59, 59);
        return (!start || eDate >= start) && (!end || eDate <= end);
    }) || [];

    // Aggregate by Category
    const data = categories?.map(c => {
        const total = filteredExpenses
            .filter(e => e.category_id === c.id)
            .reduce((sum, e) => sum + e.amount, 0);
        return { name: c.name, amount: total };
    }) || [];

    // Add "Uncategorized" if any
    const uncategorizedTotal = filteredExpenses
        .filter(e => !e.category_id || !categories?.find(c => c.id === e.category_id))
        .reduce((sum, e) => sum + e.amount, 0);

    if (uncategorizedTotal > 0) {
        data.push({ name: 'Uncategorized', amount: uncategorizedTotal });
    }

    const totalExpense = filteredExpenses.reduce((sum, e) => sum + e.amount, 0);

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Expense Report</h2>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input type="date" className="form-control" value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
                    <span style={{ alignSelf: 'center' }}>-</span>
                    <input type="date" className="form-control" value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
                </div>
            </div>

            <div style={{ height: '300px', marginBottom: '30px' }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip formatter={(value) => formatPrice(value)} />
                        <Legend />
                        <Bar dataKey="amount" fill="#8884d8" name="Amount" />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="card">
                <h3>Summary by Category</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
                    <thead>
                        <tr style={{ background: '#f8f9fa', textAlign: 'left' }}>
                            <th style={{ padding: '10px' }}>Category</th>
                            <th style={{ padding: '10px', textAlign: 'right' }}>Total Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((d, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                                <td style={{ padding: '10px' }}>{d.name}</td>
                                <td style={{ padding: '10px', textAlign: 'right' }}>{formatPrice(d.amount)}</td>
                            </tr>
                        ))}
                        <tr style={{ fontWeight: 'bold', background: '#e1f5fe' }}>
                            <td style={{ padding: '10px' }}>Total</td>
                            <td style={{ padding: '10px', textAlign: 'right' }}>{formatPrice(totalExpense)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ExpenseReport;
