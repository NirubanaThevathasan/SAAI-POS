import React, { useState, useEffect } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';

import { useSettings } from '../../context/SettingsContext';

const RegisterShift = ({ onShiftOpen }) => {
    const { formatPrice } = useSettings();
    // Check for open register
    const openRegister = useLiveQuery(() => db.registers.where('status').equals('open').first());

    const [cashInfo, setCashInfo] = useState({ open_cash: '', close_cash: '', note: '' });

    const handleOpenShift = async () => {
        if (!cashInfo.open_cash) return alert('Please enter opening cash amount');
        await db.registers.add({
            open_time: new Date(),
            open_cash: parseFloat(cashInfo.open_cash),
            status: 'open',
            user_id: 1 // TODO: Real user ID
        });
        setCashInfo({ ...cashInfo, open_cash: '' });
    };

    const handleCloseShift = async () => {
        if (!cashInfo.close_cash) return alert('Please enter closing cash amount');

        const closeTime = new Date();

        // Calculate Total Sales during this shift
        const shiftSales = await db.sales
            .where('date').between(new Date(openRegister.open_time), closeTime, true, true)
            .toArray();

        const totalSales = shiftSales.reduce((sum, sale) => sum + (sale.grandTotal || sale.total || 0), 0);
        const totalReturnRefunds = await db.returns
            .where('date').between(new Date(openRegister.open_time), closeTime, true, true)
            .filter(r => r.type === 'sale')
            .toArray()
            .then(returns => returns.reduce((sum, ret) => sum + (ret.grandTotal || ret.total || ret.totalRefund || 0), 0));

        await db.registers.update(openRegister.id, {
            close_time: closeTime,
            close_cash: parseFloat(cashInfo.close_cash),
            total_sales: totalSales - totalReturnRefunds, // Net sales
            gross_sales: totalSales,
            total_returns: totalReturnRefunds,
            status: 'closed',
            note: cashInfo.note
        });
        setCashInfo({ ...cashInfo, close_cash: '', note: '' });
    };

    if (openRegister) {
        return (
            <div className="card" style={{ background: '#e8f8f5', marginBottom: '0', padding: '15px 24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
                    <div>
                        <strong>Shift Open</strong> since {new Date(openRegister.open_time).toLocaleTimeString()}
                        <br /><small>Opening Cash: {formatPrice(openRegister.open_cash)}</small>
                    </div>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            type="number"
                            placeholder="Closing Cash"
                            value={cashInfo.close_cash}
                            onChange={e => setCashInfo({ ...cashInfo, close_cash: e.target.value })}
                            style={{ padding: '5px', width: '120px' }}
                        />
                        <input
                            type="text"
                            placeholder="Note"
                            value={cashInfo.note}
                            onChange={e => setCashInfo({ ...cashInfo, note: e.target.value })}
                            style={{ padding: '5px' }}
                        />
                        <button className="btn btn-warning" onClick={handleCloseShift}>Close Register</button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="card" style={{
            background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
            marginBottom: '15px',
            padding: '20px 24px',
            border: '2px solid #f59e0b',
            borderRadius: '12px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
                <div style={{ flex: 1 }}>
                    <h3 style={{ margin: '0 0 8px 0', color: '#92400e', fontSize: '1.1rem', fontWeight: '700' }}>⚠️ Register Not Open</h3>
                    <p style={{ margin: '0', color: '#78350f', fontSize: '0.9rem' }}>Sales and access to the POS are disabled until a shift is started.</p>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <input
                        type="number"
                        className="form-control"
                        placeholder="Cash in Hand"
                        value={cashInfo.open_cash}
                        onChange={e => setCashInfo({ ...cashInfo, open_cash: e.target.value })}
                        style={{ width: '150px', padding: '8px 12px' }}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && cashInfo.open_cash) {
                                handleOpenShift();
                            }
                        }}
                    />
                    <button className="btn btn-primary" onClick={handleOpenShift} style={{ padding: '8px 20px', fontWeight: '600' }}>
                        Open Register
                    </button>
                </div>
            </div>
        </div>
    );
};

export default RegisterShift;
