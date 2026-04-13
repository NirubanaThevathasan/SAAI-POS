import React from 'react';
import { db } from '../../db';

import { useLiveQuery } from 'dexie-react-hooks';

const AlertsReport = () => {
    const data = useLiveQuery(async () => {
        const products = await db.products.toArray();
        const batches = await db.batches.toArray();
        const today = new Date();
        const thirtyDaysFromNow = new Date();
        thirtyDaysFromNow.setDate(today.getDate() + 30);

        const lowStockProducts = products.filter(p => p.stock_quantity <= (p.alert_quantity || 5));

        const mainExpiryAlerts = products
            .filter(p => p.expiry_date && new Date(p.expiry_date) <= thirtyDaysFromNow)
            .map(p => ({
                id: `p-${p.id}`,
                product_name: p.name,
                batch_no: 'N/A',
                expiry_date: p.expiry_date,
                is_expired: new Date(p.expiry_date) < today
            }));

        const productMap = new Map(products.map(p => [p.id, p]));

        const batchExpiryAlerts = batches.filter(b => b.expiry_date && new Date(b.expiry_date) <= thirtyDaysFromNow)
            .map(b => ({
                id: `b-${b.id}`,
                product_name: productMap.get(b.product_id)?.name || 'Unknown',
                batch_no: b.batch_no || 'Unknown',
                expiry_date: b.expiry_date,
                is_expired: new Date(b.expiry_date) < today
            }));

        return {
            lowStockProducts,
            expiringBatches: [...mainExpiryAlerts, ...batchExpiryAlerts]
        };

    });

    if (!data) return <div>Loading...</div>;
    const { lowStockProducts, expiringBatches } = data;

    return (
        <div>
            <h3>Alerts Report</h3>

            <div style={{ marginBottom: '30px' }}>
                <h4 style={{ color: '#e67e22' }}>Low Stock Alerts ({lowStockProducts.length})</h4>
                <div className="card">
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
                                <th style={{ padding: '8px' }}>Product</th>
                                <th style={{ padding: '8px' }}>Code</th>
                                <th style={{ padding: '8px' }}>Current Stock</th>
                                <th style={{ padding: '8px' }}>Alert Level</th>
                            </tr>
                        </thead>
                        <tbody>
                            {lowStockProducts.map(p => (
                                <tr key={p.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                    <td style={{ padding: '8px' }}>{p.name}</td>
                                    <td style={{ padding: '8px' }}>{p.code}</td>
                                    <td style={{ padding: '8px', color: 'red', fontWeight: 'bold' }}>{p.stock_quantity}</td>
                                    <td style={{ padding: '8px' }}>{p.alert_quantity || 5}</td>
                                </tr>
                            ))}
                            {lowStockProducts.length === 0 && <tr><td colSpan="4" style={{ padding: '15px', textAlign: 'center' }}>No low stock items.</td></tr>}
                        </tbody>
                    </table>
                </div>
            </div>

            <div>
                <h4 style={{ color: '#c0392b' }}>Expiry Alerts (Next 30 Days) - ({expiringBatches.length})</h4>
                <div className="card">
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
                                <th style={{ padding: '8px' }}>Product</th>
                                <th style={{ padding: '8px' }}>Batch No</th>
                                <th style={{ padding: '8px' }}>Expiry Date</th>
                                <th style={{ padding: '8px' }}>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {expiringBatches.map(b => (
                                <tr key={b.id} style={{ borderBottom: '1px solid #f9f9f9', background: b.is_expired ? '#ffe6e6' : 'transparent' }}>
                                    <td style={{ padding: '8px' }}>{b.product_name}</td>
                                    <td style={{ padding: '8px' }}>{b.batch_no}</td>
                                    <td style={{ padding: '8px' }}>{b.expiry_date}</td>
                                    <td style={{ padding: '8px' }}>
                                        {b.is_expired ? <span style={{ color: 'red', fontWeight: 'bold' }}>EXPIRED</span> : <span style={{ color: 'orange' }}>Expiring Soon</span>}
                                    </td>
                                </tr>
                            ))}
                            {expiringBatches.length === 0 && <tr><td colSpan="4" style={{ padding: '15px', textAlign: 'center' }}>No expiring items.</td></tr>}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default AlertsReport;
