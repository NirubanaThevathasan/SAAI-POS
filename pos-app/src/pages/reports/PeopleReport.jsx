import React, { useState } from 'react';
import { db } from '../../db';

import { useLiveQuery } from 'dexie-react-hooks';
import { useSettings } from '../../context/SettingsContext';

const PeopleReport = () => {
    const { formatPrice } = useSettings();
    const data = useLiveQuery(async () => {
        const sales = await db.sales.toArray();
        const customers = await db.customers.toArray();
        const customerMap = new Map(customers.map(c => [c.id, c.name]));
        const customerSales = {};

        sales.forEach(s => {
            const name = s.customer_id ? (customerMap.get(s.customer_id) || 'Unknown') : 'Walk-in Customer';
            customerSales[name] = (customerSales[name] || 0) + (s.grandTotal || s.total || 0);
        });

        const topCustomers = Object.entries(customerSales)
            .map(([name, total]) => ({ name, total }))
            .sort((a, b) => b.total - a.total).slice(0, 5);

        const purchases = await db.purchases.toArray();
        const suppliers = await db.suppliers.toArray();
        const supplierMap = new Map(suppliers.map(s => [s.id, s.name]));
        const supplierPurchases = {};

        purchases.forEach(p => {
            const name = p.supplier_id ? (supplierMap.get(p.supplier_id) || 'Unknown') : 'Unknown';
            supplierPurchases[name] = (supplierPurchases[name] || 0) + (p.grandTotal || 0);
        });

        const topSuppliers = Object.entries(supplierPurchases)
            .map(([name, total]) => ({ name, total }))
            .sort((a, b) => b.total - a.total).slice(0, 5);

        return { topCustomers, topSuppliers };
    });

    if (!data) return <div>Loading...</div>;
    const { topCustomers, topSuppliers } = data;

    return (
        <div>
            <h3>People Report</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div className="card">
                    <h4>Top Customers (by Revenue)</h4>
                    <table style={{ width: '100%' }}>
                        <tbody>
                            {topCustomers.map((c, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                                    <td style={{ padding: '8px' }}>{i + 1}. {c.name}</td>
                                    <td style={{ padding: '8px', textAlign: 'right' }}>{formatPrice(c.total)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="card">
                    <h4>Top Suppliers (by Spend)</h4>
                    <table style={{ width: '100%' }}>
                        <tbody>
                            {topSuppliers.map((s, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                                    <td style={{ padding: '8px' }}>{i + 1}. {s.name}</td>
                                    <td style={{ padding: '8px', textAlign: 'right' }}>{formatPrice(s.total)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default PeopleReport;
