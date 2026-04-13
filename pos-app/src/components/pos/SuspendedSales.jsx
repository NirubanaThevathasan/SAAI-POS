import React from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { FaPlay, FaTrash } from 'react-icons/fa';
import { useSettings } from '../../context/SettingsContext';

const SuspendedSales = ({ onResume, onClose }) => {
    const { formatPrice } = useSettings();
    const suspended = useLiveQuery(() => db.suspended_sales.toArray());

    const handleResume = async (sale) => {
        onResume(sale);
        await db.suspended_sales.delete(sale.id);
        onClose();
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete this suspended sale?')) {
            await db.suspended_sales.delete(id);
        }
    };

    return (
        <div className="overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1100, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <div className="card" style={{ width: '600px', maxHeight: '80vh', overflowY: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                    <h3>Suspended Sales</h3>
                    <button className="btn" onClick={onClose}>Close</button>
                </div>

                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
                            <th style={{ padding: '10px' }}>Date</th>
                            <th style={{ padding: '10px' }}>Customer</th>
                            <th style={{ padding: '10px' }}>Items</th>
                            <th style={{ padding: '10px' }}>Total</th>
                            <th style={{ padding: '10px' }}>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {suspended?.map(sale => (
                            <tr key={sale.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{new Date(sale.date).toLocaleString()}</td>
                                <td style={{ padding: '10px' }}>{sale.customer_name || 'Walk-in'}</td>
                                <td style={{ padding: '10px' }}>{sale.items.length}</td>
                                <td style={{ padding: '10px', fontWeight: '800' }}>{formatPrice(sale.total)}</td>
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" title="Resume" onClick={() => handleResume(sale)} style={{ color: 'green', marginRight: '10px' }}>
                                        <FaPlay />
                                    </button>
                                    <button className="btn-icon" title="Delete" onClick={() => handleDelete(sale.id)} style={{ color: 'red' }}>
                                        <FaTrash />
                                    </button>
                                </td>
                            </tr>
                        ))}
                        {suspended?.length === 0 && <tr><td colSpan="5" style={{ padding: '20px', textAlign: 'center' }}>No suspended sales.</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default SuspendedSales;
