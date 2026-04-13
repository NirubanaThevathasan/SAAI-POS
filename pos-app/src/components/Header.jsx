import React, { useState, useEffect } from 'react';
import NotificationCenter from './NotificationCenter';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useAuth } from '../context/AuthContext';
import { useSettings } from '../context/SettingsContext';
import { getLastBackupTime } from '../services/AutoBackupService';
import { FaSignOutAlt, FaUserCircle, FaPalette, FaExpand, FaCompress, FaCheckCircle } from 'react-icons/fa';

const Header = () => {
    const { user, logout } = useAuth();
    const { currency, changeCurrency, currencies, changeThemeColor, settings, formatPrice } = useSettings();
    const [lastBackup, setLastBackup] = useState(null);
    const [showThemeMenu, setShowThemeMenu] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

    const shiftStats = useLiveQuery(async () => {
        const openRegister = await db.registers.where('status').equals('open').first();
        if (!openRegister) return null;

        const startTime = new Date(openRegister.open_time);

        // Calculate Total Sales since shift start
        const sales = await db.sales
            .where('date').aboveOrEqual(startTime)
            .toArray();
        const totalSales = sales.reduce((sum, sale) => sum + (sale.grandTotal || sale.total || 0), 0);

        // Calculate Returns since shift start
        const returns = await db.returns
            .where('date').aboveOrEqual(startTime)
            .filter(r => r.type === 'sale')
            .toArray();
        const totalReturns = returns.reduce((sum, ret) => sum + (ret.grandTotal || ret.total || ret.totalRefund || 0), 0);

        // Calculate Expenses since shift start
        const expenses = await db.expenses
            .where('date').aboveOrEqual(startTime)
            .toArray();
        const totalExpenses = expenses.reduce((sum, exp) => sum + (parseFloat(exp.amount) || 0), 0);

        return { sales: totalSales, returns: totalReturns, expenses: totalExpenses };
    }, []);

    useEffect(() => {
        const updateBackupTime = () => setLastBackup(getLastBackupTime());
        updateBackupTime();
        const interval = setInterval(updateBackupTime, 60000);
        return () => clearInterval(interval);
    }, []);

    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
            setIsFullscreen(true);
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                setIsFullscreen(false);
            }
        }
    };

    const themeColors = [
        { name: 'Indigo', color: '#6366f1' },
        { name: 'Emerald', color: '#10b981' },
        { name: 'Rose', color: '#f43f5e' },
        { name: 'Amber', color: '#f59e0b' },
        { name: 'Blue', color: '#3b82f6' },
        { name: 'Violet', color: '#8b5cf6' },
        { name: 'Dark', color: '#1e293b' }
    ];

    // Determine background based on theme for glass effect
    const headerBg = settings.theme === 'dark' ? 'rgba(30, 41, 59, 0.8)' : 'rgba(255, 255, 255, 0.8)';
    const headerBorder = settings.theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    return (
        <header className="header" style={{
            height: '64px',
            background: headerBg,
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            borderBottom: `1px solid ${headerBorder}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            position: 'sticky',
            top: 0,
            zIndex: 100,
            transition: 'all 0.3s'
        }}>
            {/* Left: Welcome & Status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div>
                    <h3 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: '800', background: 'linear-gradient(45deg, var(--primary-color), var(--accent-color))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            Dashboard
                        </span>
                        <span style={{ color: 'var(--border-color)' }}>/</span>
                        <span style={{ fontWeight: '400', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Overview</span>
                    </h3>
                </div>
                {lastBackup && (
                    <span style={{
                        fontSize: '0.75rem', color: '#10b981', background: 'rgba(16, 185, 129, 0.1)',
                        padding: '3px 10px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '4px',
                        fontWeight: '600'
                    }} title={`Last backup: ${lastBackup}`}>
                        <FaCheckCircle /> Saved
                    </span>
                )}
            </div>

            {/* Middle: Shift Stats */}
            {shiftStats && (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '24px',
                    background: 'var(--card-bg)',
                    padding: '6px 16px',
                    borderRadius: '12px',
                    border: '1px solid var(--border-color)',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1 }}>
                        <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: '600' }}>Sales</span>
                        <span style={{ fontSize: '0.95rem', fontWeight: '700', color: '#10b981' }}>{formatPrice(shiftStats.sales)}</span>
                    </div>
                    <div style={{ width: '1px', height: '20px', background: 'var(--border-color)' }}></div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1 }}>
                        <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: '600' }}>Returns</span>
                        <span style={{ fontSize: '0.95rem', fontWeight: '700', color: '#ef4444' }}>{formatPrice(shiftStats.returns)}</span>
                    </div>
                    <div style={{ width: '1px', height: '20px', background: 'var(--border-color)' }}></div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1 }}>
                        <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: '600' }}>Expenses</span>
                        <span style={{ fontSize: '0.95rem', fontWeight: '700', color: '#f59e0b' }}>{formatPrice(shiftStats.expenses)}</span>
                    </div>
                </div>
            )}

            {/* Right: Actions */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>

                <button onClick={toggleFullscreen} className="btn-icon">
                    {isFullscreen ? <FaCompress /> : <FaExpand />}
                </button>

                <div style={{ position: 'relative' }}>
                    <button
                        onClick={() => setShowThemeMenu(!showThemeMenu)}
                        className="btn-icon"
                        style={{ color: settings.primary_color }}
                    >
                        <FaPalette />
                    </button>
                    {showThemeMenu && (
                        <div className="card floating-menu" style={{
                            position: 'absolute', top: '120%', right: 0,
                            padding: '16px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px',
                            minWidth: '160px', zIndex: 101
                        }}>
                            {themeColors.map(tc => (
                                <div
                                    key={tc.color}
                                    onClick={() => { changeThemeColor(tc.color); setShowThemeMenu(false); }}
                                    style={{
                                        width: '28px', height: '28px', borderRadius: '50%',
                                        background: tc.color, cursor: 'pointer',
                                        boxShadow: settings.primary_color === tc.color ? `0 0 0 3px var(--bg-color), 0 0 0 5px ${tc.color}` : 'none',
                                        transition: 'transform 0.2s'
                                    }}
                                    className="color-swatch"
                                />
                            ))}
                        </div>
                    )}
                </div>

                <div className="select-wrapper" style={{ background: 'var(--bg-color)', borderRadius: '10px', padding: '6px 12px', border: '1px solid var(--border-color)' }}>
                    <select
                        value={currency.code}
                        onChange={(e) => changeCurrency(e.target.value)}
                        style={{ border: 'none', background: 'transparent', color: 'var(--text-primary)', outline: 'none', fontSize: '0.85rem', fontWeight: '700' }}
                    >
                        {currencies.map(c => <option key={c.code} value={c.code}>{c.code}</option>)}
                    </select>
                </div>

                <NotificationCenter />

                <div style={{ width: '1px', height: '32px', background: 'var(--border-color)' }}></div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ textAlign: 'right', lineHeight: '1.2' }}>
                        <div style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-primary)' }}>{user?.username}</div>
                        <span style={{ fontSize: '0.7rem', color: 'var(--primary-color)', background: 'rgba(99, 102, 241, 0.1)', padding: '2px 6px', borderRadius: '4px', textTransform: 'uppercase', fontWeight: '600' }}>{user?.role}</span>
                    </div>
                    <div style={{ position: 'relative' }}>
                        <FaUserCircle style={{ fontSize: '40px', color: 'var(--text-secondary)', opacity: 0.8 }} />
                        <div style={{ position: 'absolute', bottom: 2, right: 2, width: '12px', height: '12px', background: '#10b981', borderRadius: '50%', border: '2px solid var(--card-bg)' }}></div>
                    </div>
                    <button
                        onClick={logout}
                        style={{
                            border: 'none', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444',
                            width: '36px', height: '36px', borderRadius: '10px', cursor: 'pointer',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s'
                        }}
                        title="Sign Out"
                    >
                        <FaSignOutAlt />
                    </button>
                </div>
            </div>

            <style>{`
                .btn-icon {
                    background: transparent;
                    border: none;
                    color: var(--text-secondary);
                    font-size: 1.2rem;
                    cursor: pointer;
                    width: 40px; height: 40px;
                    border-radius: 10px;
                    display: flex; alignItems: center; justifyContent: center;
                    transition: all 0.2s;
                }
                .btn-icon:hover {
                    background: var(--bg-color);
                    color: var(--primary-color);
                    transform: translateY(-2px);
                }
                .color-swatch:hover {
                    transform: scale(1.2);
                }
                .floating-menu {
                    animation: slideDown 0.2s ease-out;
                }
                @keyframes slideDown {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </header>
    );
};

export default Header;
