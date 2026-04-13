import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';
import { FaTruck, FaEdit, FaTrash, FaCheckCircle, FaBox, FaMapMarkerAlt, FaSearch, FaFilter, FaShippingFast } from 'react-icons/fa';

const Deliveries = () => {
    const { formatPrice } = useSettings();
    const deliveries = useLiveQuery(() => db.deliveries.orderBy('date').reverse().toArray());
    const customers = useLiveQuery(() => db.customers.toArray());

    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');

    const filteredDeliveries = deliveries?.filter(d => {
        const customer = customers?.find(c => c.id === d.customer_id);
        const matchesSearch = searchTerm === '' ||
            (d.reference_no?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                customer?.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                d.sale_id.toString().includes(searchTerm));
        const matchesStatus = statusFilter === 'all' || d.status === statusFilter;
        return matchesSearch && matchesStatus;
    });

    const updateStatus = async (id, newStatus) => {
        if (newStatus === 'delivering') {
            const d = deliveries.find(x => x.id === id);
            if (!d.reference_no) return alert('Please assign a tracking number before marking as delivering');
        }
        await db.deliveries.update(id, { status: newStatus });
    };

    const updateTracking = async (id, trackingNo) => {
        await db.deliveries.update(id, { reference_no: trackingNo });
    };

    const deleteDelivery = async (id) => {
        if (window.confirm('Are you sure you want to delete this delivery record?')) {
            await db.deliveries.delete(id);
        }
    };

    const getStatusStyle = (status) => {
        switch (status) {
            case 'packing': return { background: 'rgba(245, 158, 11, 0.1)', color: '#d97706', label: 'Packing', icon: <FaBox /> };
            case 'delivering': return { background: 'rgba(59, 130, 246, 0.1)', color: '#2563eb', label: 'Delivering', icon: <FaTruck /> };
            case 'delivered': return { background: 'rgba(16, 185, 129, 0.1)', color: '#059669', label: 'Delivered', icon: <FaCheckCircle /> };
            case 'cancelled': return { background: 'rgba(244, 63, 94, 0.1)', color: '#e11d48', label: 'Cancelled', icon: <FaTrash /> };
            default: return { background: 'var(--bg-secondary)', color: 'var(--text-secondary)', label: status, icon: <FaShippingFast /> };
        }
    };

    return (
        <div style={{ animation: 'fadeIn 0.4s ease-out' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                <div>
                    <h2 style={{ fontSize: '1.75rem', fontWeight: '800', letterSpacing: '-0.025em', marginBottom: '4px' }}>Delivery Management</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Track and manage order shipments from warehouse to doorstep</p>
                </div>
                <div style={{
                    background: 'var(--glass-bg)',
                    padding: '8px 16px',
                    borderRadius: '12px',
                    border: '1px solid var(--glass-border)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px'
                }}>
                    <FaShippingFast style={{ color: 'var(--primary-color)' }} />
                    <span style={{ fontWeight: '700', color: 'var(--text-primary)' }}>{deliveries?.length || 0} Total Deliveries</span>
                </div>
            </div>

            <div className="card" style={{ padding: '24px', marginBottom: '24px', background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', backdropFilter: 'blur(10px)' }}>
                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '300px', position: 'relative' }}>
                        <FaSearch style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                        <input
                            type="text"
                            className="form-control"
                            placeholder="Search by sale ref, tracking # or customer..."
                            style={{ paddingLeft: '48px', height: '48px', borderRadius: '12px', background: '#fff' }}
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <div style={{ width: '200px', position: 'relative' }}>
                        <FaFilter style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                        <select
                            className="form-control"
                            style={{ paddingLeft: '48px', height: '48px', borderRadius: '12px', background: '#fff' }}
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                        >
                            <option value="all">All Statuses</option>
                            <option value="packing">Packing</option>
                            <option value="delivering">Delivering</option>
                            <option value="delivered">Delivered</option>
                            <option value="cancelled">Cancelled</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="card" style={{ padding: '0', overflow: 'hidden', borderRadius: '16px', border: '1px solid var(--border-color)' }}>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ background: '#f8fafc', borderBottom: '1px solid #f1f5f9' }}>
                                <th style={{ padding: '16px 24px', textAlign: 'left', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Shipment Date</th>
                                <th style={{ padding: '16px 24px', textAlign: 'left', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Sale ID</th>
                                <th style={{ padding: '16px 24px', textAlign: 'left', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Customer & Destination</th>
                                <th style={{ padding: '16px 24px', textAlign: 'left', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Status</th>
                                <th style={{ padding: '16px 24px', textAlign: 'left', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Tracking / Reference</th>
                                <th style={{ padding: '16px 24px', textAlign: 'right', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredDeliveries?.map((delivery) => {
                                const customer = customers?.find(c => c.id === delivery.customer_id);
                                const st = getStatusStyle(delivery.status);
                                return (
                                    <React.Fragment key={delivery.id}>
                                        <tr style={{ borderBottom: '1px solid #f1f5f9', transition: 'background 0.2s', background: '#fff' }}>
                                            <td style={{ padding: '16px 24px', fontSize: '0.9rem', color: '#1e293b' }}>
                                                <div style={{ fontWeight: '600' }}>{new Date(delivery.date).toLocaleDateString()}</div>
                                                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{new Date(delivery.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                                            </td>
                                            <td style={{ padding: '16px 24px' }}>
                                                <span style={{
                                                    display: 'inline-flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    padding: '4px 10px',
                                                    borderRadius: '8px',
                                                    background: 'var(--bg-secondary)',
                                                    fontSize: '0.85rem',
                                                    fontWeight: '700',
                                                    color: 'var(--primary-color)'
                                                }}>
                                                    #{delivery.sale_id}
                                                </span>
                                            </td>
                                            <td style={{ padding: '16px 24px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                                    <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--primary-hover)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: '800' }}>
                                                        {customer?.name?.charAt(0) || 'W'}
                                                    </div>
                                                    <div>
                                                        <div style={{ fontSize: '0.9rem', fontWeight: '700', color: '#1e293b' }}>{customer?.name || 'Walk-in'}</div>
                                                        <div style={{ fontSize: '0.8rem', color: '#64748b' }}>{customer?.phone || 'No Phone'}</div>
                                                    </div>
                                                </div>
                                                <div style={{ fontSize: '0.8rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <FaMapMarkerAlt style={{ fontSize: '0.7rem' }} />
                                                    {delivery.address || customer?.address || 'No address provided'}
                                                </div>
                                            </td>
                                            <td style={{ padding: '16px 24px' }}>
                                                <span style={{
                                                    display: 'inline-flex',
                                                    alignItems: 'center',
                                                    gap: '6px',
                                                    padding: '6px 12px',
                                                    borderRadius: '20px',
                                                    fontSize: '0.75rem',
                                                    fontWeight: '800',
                                                    background: st.background,
                                                    color: st.color,
                                                    textTransform: 'uppercase',
                                                    letterSpacing: '0.025em'
                                                }}>
                                                    {st.icon} {st.label}
                                                </span>
                                            </td>
                                            <td style={{ padding: '16px 24px' }}>
                                                <input
                                                    type="text"
                                                    className="form-control"
                                                    style={{
                                                        fontSize: '0.85rem',
                                                        padding: '4px 8px',
                                                        borderRadius: '6px',
                                                        border: '1px solid #e2e8f0',
                                                        width: '180px',
                                                        fontFamily: 'monospace'
                                                    }}
                                                    placeholder="Assign Tracking #"
                                                    value={delivery.reference_no || ''}
                                                    onChange={(e) => updateTracking(delivery.id, e.target.value)}
                                                />
                                            </td>
                                            <td style={{ padding: '16px 24px', textAlign: 'right' }}>
                                                <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                                                    <button
                                                        className="btn-icon"
                                                        style={{ color: '#059669', background: 'rgba(16, 185, 129, 0.1)' }}
                                                        onClick={() => updateStatus(delivery.id, 'delivered')}
                                                        title="Mark as Delivered"
                                                    >
                                                        <FaCheckCircle />
                                                    </button>
                                                    <button
                                                        className="btn-icon"
                                                        style={{ color: '#2563eb', background: 'rgba(59, 130, 246, 0.1)' }}
                                                        onClick={() => updateStatus(delivery.id, 'delivering')}
                                                        title="Mark as Delivering"
                                                    >
                                                        <FaTruck />
                                                    </button>
                                                    <button
                                                        className="btn-icon"
                                                        style={{ color: '#d97706', background: 'rgba(245, 158, 11, 0.1)' }}
                                                        onClick={() => updateStatus(delivery.id, 'packing')}
                                                        title="Mark as Packing"
                                                    >
                                                        <FaBox />
                                                    </button>
                                                    <button
                                                        className="btn-icon"
                                                        style={{ color: '#e11d48', background: 'rgba(244, 63, 94, 0.1)' }}
                                                        onClick={() => deleteDelivery(delivery.id)}
                                                        title="Delete Record"
                                                    >
                                                        <FaTrash />
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                        {delivery.note && (
                                            <tr key={`note-${delivery.id}`}>
                                                <td colSpan="6" style={{ padding: '0 24px 16px' }}>
                                                    <div style={{ padding: '8px 16px', background: '#fffbeb', borderLeft: '4px solid #f59e0b', borderRadius: '4px', fontSize: '0.8rem', color: '#92400e' }}>
                                                        <strong>Delivery Note:</strong> {delivery.note}
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                );
                            })}
                            {filteredDeliveries?.length === 0 && (
                                <tr>
                                    <td colSpan="6" style={{ padding: '64px', textAlign: 'center', color: '#94a3b8' }}>
                                        <FaTruck style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.1 }} />
                                        <h3 style={{ color: '#64748b', marginBottom: '8px' }}>No Deliveries Found</h3>
                                        <p style={{ maxWidth: '300px', margin: '0 auto' }}>Filter your search or process a new sale and mark it for delivery in the POS terminal.</p>
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default Deliveries;
