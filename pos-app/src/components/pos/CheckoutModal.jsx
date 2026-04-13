import React, { useState, useEffect } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';

const CheckoutModal = ({ total, onConfirm, onCancel }) => {
    const { formatPrice } = useSettings();
    // Payment Methods
    const methods = useLiveQuery(() => db.payment_methods.toArray());

    // Split Payment State
    const [payments, setPayments] = useState([{ method: 'Cash', amount: '' }]);
    const [receivedAmount, setReceivedAmount] = useState(0);
    const [isDelivery, setIsDelivery] = useState(false);
    const [shippingDetails, setShippingDetails] = useState({ address: '', notes: '' });

    // Calculate total paid
    useEffect(() => {
        const sum = payments.reduce((acc, p) => acc + (parseFloat(p.amount) || 0), 0);
        setReceivedAmount(sum);
    }, [payments]);

    const handleAddPayment = () => {
        setPayments([...payments, { method: 'Cash', amount: '' }]);
    };

    const handleRemovePayment = (index) => {
        setPayments(payments.filter((_, i) => i !== index));
    };

    const handlePaymentChange = (index, field, value) => {
        const newPayments = [...payments];
        if (field === 'amount') {
            const num = parseFloat(value);
            if (num < 0) return; // Prevent negative entry
            newPayments[index][field] = value;
        } else {
            newPayments[index][field] = value;
        }
        setPayments(newPayments);
    };

    const balance = receivedAmount - total;
    const isPaid = receivedAmount >= total;

    const handleSubmit = () => {
        if (!isPaid) return alert('Insufficient payment amount');

        // Filter out empty rows
        const validPayments = payments.filter(p => p.amount > 0);

        onConfirm({
            method: validPayments.length > 1 ? 'Split' : validPayments[0]?.method || 'Cash',
            payments: validPayments,
            received: receivedAmount,
            balance: balance > 0 ? balance : 0, // Change to return
            isDelivery: isDelivery,
            shippingDetails: isDelivery ? shippingDetails : null
        });
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
        }}>
            <div className="modal-content card" style={{ padding: '20px', width: '500px', background: 'white' }}>
                <h2 className="mb-3">Checkout</h2>

                <div style={{ textAlign: 'center', marginBottom: '20px', background: '#f8f9fa', padding: '15px', borderRadius: '12px' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Total Payable</h4>
                    <h1 style={{ color: 'var(--success-color)', fontWeight: '800' }}>{formatPrice(total)}</h1>
                </div>

                <div className="payment-rows" style={{ maxHeight: '200px', overflowY: 'auto', marginBottom: '15px' }}>
                    {payments.map((p, i) => (
                        <div key={i} style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                            <select
                                className="form-control"
                                value={p.method}
                                onChange={e => handlePaymentChange(i, 'method', e.target.value)}
                                style={{ flex: 1 }}
                            >
                                <option>Cash</option>
                                <option>Card</option>
                                <option>UPI</option>
                                <option>Gift Card</option>
                                <option>Cheque</option>
                                {methods?.map(m => <option key={m.id} value={m.name}>{m.name}</option>)}
                            </select>
                            <input
                                type="number"
                                className="form-control"
                                placeholder="Amount"
                                value={p.amount}
                                onChange={e => handlePaymentChange(i, 'amount', e.target.value)}
                                style={{ flex: 1 }}
                            />
                            {payments.length > 1 && (
                                <button type="button" className="btn btn-danger" onClick={() => handleRemovePayment(i)}>X</button>
                            )}
                        </div>
                    ))}
                </div>

                <button type="button" className="btn btn-link mb-3" onClick={handleAddPayment}>+ Add Another Payment Method</button>

                <div style={{
                    marginBottom: '15px',
                    padding: '12px',
                    background: isDelivery ? 'rgba(99, 102, 241, 0.05)' : '#f8f9fa',
                    borderRadius: '12px',
                    border: isDelivery ? '1px solid var(--primary-color)' : '1px solid transparent',
                    transition: 'all 0.3s'
                }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', marginBottom: isDelivery ? '12px' : '0' }}>
                        <input
                            type="checkbox"
                            style={{ width: '18px', height: '18px' }}
                            checked={isDelivery}
                            onChange={e => setIsDelivery(e.target.checked)}
                        />
                        <span style={{ fontWeight: '700', color: isDelivery ? 'var(--primary-color)' : 'var(--text-secondary)' }}>Register for Delivery / Shipment</span>
                    </label>

                    {isDelivery && (
                        <div style={{ animation: 'fadeIn 0.3s ease-out' }}>
                            <div style={{ marginBottom: '10px' }}>
                                <label style={{ fontSize: '0.8rem', fontWeight: '600', color: '#64748b' }}>Destination Address</label>
                                <textarea
                                    className="form-control"
                                    rows="2"
                                    placeholder="Enter full delivery address..."
                                    value={shippingDetails.address}
                                    onChange={e => setShippingDetails({ ...shippingDetails, address: e.target.value })}
                                    style={{ fontSize: '0.85rem' }}
                                />
                            </div>
                            <div>
                                <label style={{ fontSize: '0.8rem', fontWeight: '600', color: '#64748b' }}>Delivery Instructions / Note</label>
                                <input
                                    type="text"
                                    className="form-control"
                                    placeholder="e.g. Leave at front door"
                                    value={shippingDetails.notes}
                                    onChange={e => setShippingDetails({ ...shippingDetails, notes: e.target.value })}
                                    style={{ fontSize: '0.85rem' }}
                                />
                            </div>
                        </div>
                    )}
                </div>

                <div style={{ borderTop: '1px solid #eee', paddingTop: '15px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span>Total Paid:</span>
                        <span style={{ fontWeight: '800' }}>{formatPrice(receivedAmount)}</span>
                    </div>
                    {balance > 0 && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', color: 'var(--primary-color)' }}>
                            <span>Change Return:</span>
                            <span style={{ fontWeight: '800' }}>{formatPrice(balance)}</span>
                        </div>
                    )}
                    {balance < 0 && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', color: 'var(--accent-color)' }}>
                            <span>Balance Due:</span>
                            <span style={{ fontWeight: '800' }}>{formatPrice(Math.abs(balance))}</span>
                        </div>
                    )}
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                    <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
                    <button className="btn btn-success" onClick={handleSubmit} disabled={!isPaid}>Finalize Payment</button>
                </div>
            </div>
        </div>
    );
};

export default CheckoutModal;
