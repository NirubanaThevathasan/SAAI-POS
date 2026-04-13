import React, { useState } from 'react';
import { db } from '../../db';

import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';

const PurchaseReport = () => {
    const { formatPrice } = useSettings();
    const [dateRange, setDateRange] = useState({
        start: new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0],
        end: new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).toISOString().split('T')[0]
    });

    const data = useLiveQuery(async () => {
        if (!dateRange.start || !dateRange.end) return null;
        const allPurchases = await db.purchases.toArray();
        const start = new Date(dateRange.start);
        const end = new Date(dateRange.end);
        end.setHours(23, 59, 59);

        const relevantPurchases = allPurchases.filter(p => {
            const d = new Date(p.date);
            return d >= start && d <= end;
        });

        const totalAmount = relevantPurchases.reduce((sum, p) => sum + (p.grand_total || p.total || 0), 0);
        return { relevantPurchases, totalAmount };
    }, [dateRange]);

    if (!data) return <div>Loading...</div>;
    const { relevantPurchases, totalAmount } = data;

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3>Purchase Report</h3>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <label>From:</label>
                    <input type="date" className="form-control" value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
                    <label>To:</label>
                    <input type="date" className="form-control" value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
                </div>
            </div>

            <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
                <div className="card" style={{ flex: 1, textAlign: 'center' }}>
                    <h4>Total Purchases</h4>
                    <div className="big-number">{relevantPurchases.length}</div>
                </div>
                <div className="card" style={{ flex: 1, textAlign: 'center' }}>
                    <h4>Total Cost</h4>
                    <div className="big-number" style={{ color: '#e74c3c' }}>{formatPrice(totalAmount)}</div>
                </div>
            </div>

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Ref</th>
                            <th style={{ padding: '10px' }}>Supplier</th>
                            <th style={{ padding: '10px' }}>Status</th>
                            <th style={{ padding: '10px' }}>Payment</th>
                            <th style={{ padding: '10px' }}>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {relevantPurchases.map(p => (
                            <tr key={p.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(p.date).toLocaleDateString()}</td>
                                <td style={{ padding: '10px' }}>{p.reference_no}</td>
                                <td style={{ padding: '10px' }}>{p.supplier_id || '-'}</td>
                                <td style={{ padding: '10px' }}>{p.status}</td>
                                <td style={{ padding: '10px' }}>{p.payment_status}</td>
                                <td style={{ padding: '10px' }}>{formatPrice(p.grand_total || p.total || 0)}</td>
                            </tr>
                        ))}
                        {relevantPurchases.length === 0 && <tr><td colSpan="6" style={{ padding: '20px', textAlign: 'center' }}>No records found</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PurchaseReport;
