import React, { useEffect, useState } from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaBell, FaCheck, FaExclamationTriangle, FaInfoCircle, FaTrash } from 'react-icons/fa';

const NotificationCenter = () => {
    const notifications = useLiveQuery(() => db.notifications.reverse().sortBy('date'));
    const [isOpen, setIsOpen] = useState(false);
    const [unreadCount, setUnreadCount] = useState(0);

    useEffect(() => {
        if (notifications) {
            setUnreadCount(notifications.filter(n => !n.read).length);
        }
    }, [notifications]);

    // Automatic Alert Generation (Run once on mount or periodically)
    useEffect(() => {
        const generateAlerts = async () => {
            // 1. Low Stock Alerts
            const products = await db.products.toArray();
            const lowStock = products.filter(p => p.stock_quantity <= (p.alert_quantity || 5));

            for (const p of lowStock) {
                const existing = await db.notifications.where({ type: 'alert', related_id: p.id }).first();
                if (!existing) {
                    await db.notifications.add({
                        type: 'alert',
                        message: `Low Stock: ${p.name} is down to ${p.stock_quantity}`,
                        read: 0,
                        date: new Date().toISOString(),
                        related_id: p.id
                    });
                }
            }

            // 2. Expiry Alerts (Next 30 days)
            const today = new Date();
            const thirtyDays = new Date();
            thirtyDays.setDate(today.getDate() + 30);

            // A) Check Products Table
            for (const p of products) {
                if (p.expiry_date) {
                    const expiry = new Date(p.expiry_date);
                    if (expiry <= thirtyDays) {
                        const isExpired = expiry < today;
                        const msg = isExpired
                            ? `EXPIRED: ${p.name} expired on ${p.expiry_date}`
                            : `Expiring Soon: ${p.name} expires on ${p.expiry_date}`;

                        const existing = await db.notifications.where({ message: msg }).first();
                        if (!existing) {
                            await db.notifications.add({
                                type: 'alert',
                                message: msg,
                                read: 0,
                                date: new Date().toISOString(),
                                related_id: p.id
                            });
                        }
                    }
                }
            }

            // B) Check Batches Table
            const batches = await db.batches.toArray();
            for (const b of batches) {
                if (b.expiry_date) {
                    const expiry = new Date(b.expiry_date);
                    if (expiry <= thirtyDays) {
                        const product = await db.products.get(b.product_id);
                        const isExpired = expiry < today;
                        const msg = isExpired
                            ? `EXPIRED: ${product?.name} (Batch ${b.batch_no}) expired on ${b.expiry_date}`
                            : `Expiring Soon: ${product?.name} (Batch ${b.batch_no}) expires on ${b.expiry_date}`;

                        const existing = await db.notifications.where({ message: msg }).first();
                        if (!existing) {
                            await db.notifications.add({
                                type: 'alert',
                                message: msg,
                                read: 0,
                                date: new Date().toISOString(),
                                related_id: b.id
                            });
                        }
                    }
                }
            }

        };
        generateAlerts();

        // Polling every minute for demo purposes? Or just leave it on mount.
        // const interval = setInterval(generateAlerts, 60000);
        // return () => clearInterval(interval);
    }, []);

    const markAsRead = async (id) => {
        await db.notifications.update(id, { read: 1 });
    };

    const markAllRead = async () => {
        const unread = await db.notifications.filter(n => !n.read).toArray();
        await db.notifications.bulkPut(unread.map(n => ({ ...n, read: 1 })));
    };

    const deleteNotification = async (id, e) => {
        e.stopPropagation();
        await db.notifications.delete(id);
    };

    return (
        <div style={{ position: 'relative' }}>
            <button
                className="btn-icon"
                onClick={() => setIsOpen(!isOpen)}
                style={{ position: 'relative' }}
            >
                <FaBell style={{ fontSize: '1.2rem', color: '#555' }} />
                {unreadCount > 0 && (
                    <span style={{
                        position: 'absolute', top: '-5px', right: '-5px',
                        background: 'red', color: 'white', borderRadius: '50%',
                        padding: '2px 6px', fontSize: '10px', fontWeight: 'bold'
                    }}>
                        {unreadCount}
                    </span>
                )}
            </button>

            {isOpen && (
                <div style={{
                    position: 'absolute', top: '40px', right: '0', width: '350px',
                    background: 'white', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                    zIndex: 1000, overflow: 'hidden', border: '1px solid #eee'
                }}>
                    <div style={{ padding: '10px 15px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f8f9fa' }}>
                        <h4 style={{ margin: 0, fontSize: '14px' }}>Notifications</h4>
                        {unreadCount > 0 && (
                            <button
                                onClick={markAllRead}
                                style={{ background: 'none', border: 'none', color: '#1976d2', cursor: 'pointer', fontSize: '12px' }}
                            >
                                Mark all read
                            </button>
                        )}
                    </div>
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {notifications?.length === 0 ? (
                            <div style={{ padding: '20px', textAlign: 'center', color: '#888' }}>No notifications</div>
                        ) : (
                            notifications?.map(n => (
                                <div
                                    key={n.id}
                                    onClick={() => markAsRead(n.id)}
                                    style={{
                                        padding: '12px 15px',
                                        borderBottom: '1px solid #eee',
                                        background: n.read ? 'white' : '#f0f7ff',
                                        cursor: 'pointer',
                                        display: 'flex', gap: '10px'
                                    }}
                                >
                                    <div style={{ marginTop: '2px' }}>
                                        {n.type === 'alert' && <FaExclamationTriangle color="#e74c3c" />}
                                        {n.type === 'info' && <FaInfoCircle color="#3498db" />}
                                        {n.type === 'success' && <FaCheck color="#2ecc71" />}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <p style={{ margin: '0 0 5px 0', fontSize: '13px', color: '#333', lineHeight: '1.4' }}>{n.message}</p>
                                        <small style={{ color: '#999', fontSize: '11px' }}>{new Date(n.date).toLocaleString()}</small>
                                    </div>
                                    <div onClick={(e) => deleteNotification(n.id, e)} style={{ opacity: 0.5, cursor: 'pointer' }}>
                                        <FaTrash size={12} />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default NotificationCenter;
