import React from 'react';
import { FaTrash, FaMinus, FaPlus } from 'react-icons/fa';
import { useSettings } from '../../context/SettingsContext';
import { useAuth } from '../../context/AuthContext';

const Cart = ({ cart, onRemove, onUpdateQuantity, onPriceUpdate, onClear, onCheckout }) => {
    const { formatPrice, currency } = useSettings();
    const { hasPermission } = useAuth();
    const [discountRate, setDiscountRate] = React.useState(0);
    const [shipping, setShipping] = React.useState(0);
    const [taxRate, setTaxRate] = React.useState(0); // Default 0%

    const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    const discountAmount = subtotal * (discountRate / 100);
    const taxAmount = (subtotal - discountAmount) * (taxRate / 100);
    const grandTotal = subtotal - discountAmount + taxAmount + parseFloat(shipping || 0);

    const handleCheckoutClick = () => {
        if (cart.length === 0) return alert('Cart is empty');
        const hasInvalidQty = cart.some(item => item.quantity <= 0);
        if (hasInvalidQty) return alert('All items must have a quantity greater than 0.');

        onCheckout({
            subtotal,
            discountRate,
            discount: discountAmount,
            taxRate,
            tax: taxAmount,
            shipping: parseFloat(shipping || 0),
            total: grandTotal
        });
    };

    const [showAdjustments, setShowAdjustments] = React.useState(false);

    return (
        <div className="cart-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div className="cart-header" style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700' }}>Current Sale ({cart.length})</h3>
                {hasPermission('pos_clear_cart') && <button className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: '0.75rem' }} onClick={onClear} disabled={cart.length === 0}>Clear List</button>}
            </div>

            <div className="cart-items" style={{ flex: '1 1 auto', overflowY: 'auto', minHeight: 0, padding: '0' }}>
                {cart.map(item => (
                    <div key={item.id} className="cart-item" style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '8px 16px',
                        borderBottom: '1px solid var(--border-color)',
                        gap: '10px'
                    }}>
                        <div style={{ flex: '1', minWidth: 0 }}>
                            <div style={{ fontWeight: '600', wordBreak: 'break-word', fontSize: '0.85rem', color: 'var(--text-primary)', lineHeight: '1.2' }}>
                                {item.name}
                            </div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '2px' }}>
                                <span>{currency.symbol}</span>
                                {hasPermission('pos_price_edit') ? (
                                    <input
                                        type="number"
                                        step="0.01"
                                        style={{
                                            width: '55px',
                                            padding: '0 4px',
                                            fontSize: '0.8rem',
                                            height: '20px',
                                            border: '1px solid transparent',
                                            background: 'rgba(0,0,0,0.03)'
                                        }}
                                        value={item.price}
                                        onChange={(e) => onPriceUpdate(item.id, parseFloat(e.target.value) || 0)}
                                        onClick={(e) => e.target.select()}
                                    />
                                ) : (
                                    <span>{item.price}</span>
                                )}
                                <span style={{ marginLeft: '4px', opacity: 0.6 }}>x {item.quantity}</span>
                            </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
                            <button className="btn-icon" style={{ width: '24px', height: '24px' }} onClick={() => onUpdateQuantity(item.id, -1)} disabled={item.quantity <= 1}>
                                <FaMinus size={8} />
                            </button>
                            <input
                                type="number"
                                min="1"
                                style={{
                                    width: '35px',
                                    textAlign: 'center',
                                    fontSize: '0.85rem',
                                    fontWeight: '700',
                                    border: 'none',
                                    background: 'none',
                                    padding: '0'
                                }}
                                value={item.quantity}
                                onChange={(e) => onUpdateQuantity(item.id, parseInt(e.target.value) || 1, true)}
                                onClick={(e) => e.target.select()}
                            />
                            <button className="btn-icon" style={{ width: '24px', height: '24px' }} onClick={() => onUpdateQuantity(item.id, 1)}>
                                <FaPlus size={8} />
                            </button>
                        </div>

                        <div style={{
                            textAlign: 'right',
                            minWidth: '70px',
                            flexShrink: 0
                        }}>
                            <div style={{ fontWeight: '700', fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                                {formatPrice(item.price * item.quantity)}
                            </div>
                            <button
                                onClick={() => onRemove(item.id)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    color: 'var(--accent-color)',
                                    cursor: 'pointer',
                                    fontSize: '0.7rem',
                                    padding: '0',
                                    opacity: 0.7
                                }}
                            >
                                <FaTrash size={10} />
                            </button>
                        </div>
                    </div>
                ))}
                {cart.length === 0 && (
                    <div style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                        <p style={{ fontSize: '0.9rem' }}>Ready for a new sale</p>
                    </div>
                )}
            </div>

            <div className="cart-footer" style={{ flexShrink: 0, padding: '16px', background: '#f8fafc' }}>
                {/* Compact Stats Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    {hasPermission('pos_discount') ? (
                        <button
                            className="btn btn-link"
                            style={{ padding: 0, fontSize: '0.8rem', color: 'var(--primary-color)', textDecoration: 'none' }}
                            onClick={() => setShowAdjustments(!showAdjustments)}
                        >
                            {showAdjustments ? 'Hide Adjustments' : '+ Add Discount/Tax/Shipping'}
                        </button>
                    ) : (
                        <div />
                    )}
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Subtotal: <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{formatPrice(subtotal)}</span>
                    </div>
                </div>

                {/* Collapsible Adjustments */}
                {showAdjustments && (
                    <div style={{
                        background: '#fff',
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        marginBottom: '16px',
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, 1fr)',
                        gap: '10px'
                    }}>
                        <div>
                            <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '2px' }}>Discount %</label>
                            <input type="number" style={{ padding: '4px', height: '28px', fontSize: '0.8rem' }} value={discountRate} onChange={e => setDiscountRate(e.target.value)} />
                        </div>
                        <div>
                            <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '2px' }}>Tax %</label>
                            <input type="number" style={{ padding: '4px', height: '28px', fontSize: '0.8rem' }} value={taxRate} onChange={e => setTaxRate(e.target.value)} />
                        </div>
                        <div>
                            <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '2px' }}>Shipping</label>
                            <input type="number" style={{ padding: '4px', height: '28px', fontSize: '0.8rem' }} value={shipping} onChange={e => setShipping(e.target.value)} />
                        </div>
                    </div>
                )}

                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '16px',
                    paddingTop: showAdjustments ? '0' : '4px'
                }}>
                    <div style={{ lineHeight: 1 }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Grand Total</span>
                        <div style={{ fontSize: '1.75rem', fontWeight: '900', color: 'var(--primary-color)', letterSpacing: '-0.025em', marginTop: '2px' }}>
                            {formatPrice(grandTotal)}
                        </div>
                    </div>
                    <button className="btn btn-primary" style={{ height: '54px', padding: '0 32px', fontSize: '1.25rem', borderRadius: '14px', boxShadow: '0 8px 16px -4px rgba(99, 102, 241, 0.4)' }} onClick={handleCheckoutClick} disabled={cart.length === 0}>
                        PAY
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Cart;
