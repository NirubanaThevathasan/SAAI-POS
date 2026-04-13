import React, { useState, useEffect } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';

const RegisterRow = ({ reg }) => {
    const { formatPrice } = useSettings();

    // If total_sales is already saved and status is closed, use it. 
    // Otherwise, calculate it on the fly (for open registers OR old records).
    const calculatedSales = useLiveQuery(async () => {
        if (reg.status === 'closed' && reg.total_sales !== undefined) {
            return reg.total_sales;
        }

        const endTime = reg.close_time ? new Date(reg.close_time) : new Date();
        const sales = await db.sales
            .where('date').between(new Date(reg.open_time), endTime, true, true)
            .toArray();

        const returns = await db.returns
            .where('date').between(new Date(reg.open_time), endTime, true, true)
            .filter(r => r.type === 'sale')
            .toArray();

        const totalSalesVal = sales.reduce((sum, sale) => sum + (sale.grandTotal || sale.total || 0), 0);
        const totalReturnsVal = returns.reduce((sum, ret) => sum + (ret.grandTotal || ret.total || ret.totalRefund || 0), 0);

        return totalSalesVal - totalReturnsVal;
    }, [reg.id, reg.status, reg.close_time, reg.total_sales]);

    return (
        <tr style={{ borderBottom: '1px solid #eee' }}>
            <td style={{ padding: '10px' }}>{new Date(reg.open_time).toLocaleString()}</td>
            <td style={{ padding: '10px' }}>{reg.close_time ? new Date(reg.close_time).toLocaleString() : '-'}</td>
            <td style={{ padding: '10px' }}>
                <span style={{
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '0.85rem',
                    background: reg.status === 'open' ? 'rgba(16, 185, 129, 0.1)' : '#eee',
                    color: reg.status === 'open' ? '#059669' : '#666'
                }}>
                    {reg.status}
                </span>
            </td>
            <td style={{ padding: '10px' }}>{formatPrice(reg.open_cash || 0)}</td>
            <td style={{ padding: '10px' }}>{formatPrice(reg.close_cash || reg.open_cash || 0)}</td>
            <td style={{ padding: '10px' }}>
                {calculatedSales !== undefined ? formatPrice(calculatedSales) : '...'}
            </td>
        </tr>
    );
};

const RegisterReport = () => {
    const registers = useLiveQuery(() => db.registers.orderBy('open_time').reverse().limit(20).toArray());

    if (!registers) return <div>Loading...</div>;

    return (
        <div>
            <h3>Register Report (Shift Logs)</h3>
            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ background: '#f8f9fa', textAlign: 'left' }}>
                            <th style={{ padding: '10px' }}>Opened</th>
                            <th style={{ padding: '10px' }}>Closed</th>
                            <th style={{ padding: '10px' }}>Status</th>
                            <th style={{ padding: '10px' }}>Opening Cash</th>
                            <th style={{ padding: '10px' }}>Cash in Hand</th>
                            <th style={{ padding: '10px' }}>Net Sales</th>
                        </tr>
                    </thead>
                    <tbody>
                        {registers.map(reg => <RegisterRow key={reg.id} reg={reg} />)}
                        {registers.length === 0 && <tr><td colSpan="6" style={{ padding: '20px', textAlign: 'center' }}>No register logs found.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default RegisterReport;
