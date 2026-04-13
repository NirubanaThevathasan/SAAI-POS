
import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FaHome, FaBox, FaShoppingCart, FaChartLine, FaUsers, FaCog, FaSignOutAlt, FaTruck, FaExchangeAlt, FaUndo, FaMoneyBillWave, FaShoppingBag, FaMoon, FaSun, FaAngleRight, FaThumbsUp, FaStickyNote, FaFileInvoice, FaRobot } from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';
import { useSettings } from '../context/SettingsContext';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';

const Sidebar = () => {
    const { user, logout, hasPermission } = useAuth();
    const navigate = useNavigate();
    const { settings, toggleTheme } = useSettings();
    const unreadCount = useLiveQuery(() => db.notifications.where('read').equals(0).count());

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const menuItems = [
        { path: '/', icon: <FaHome />, label: 'Dashboard', exact: true },
        { path: '/pos', icon: <FaShoppingCart />, label: 'POS Terminal', permission: 'pos_access' },

        { header: 'Inventory' },
        { path: '/products', icon: <FaBox />, label: 'Products', permission: 'products_view' },
        { path: '/requests', icon: <FaThumbsUp />, label: 'Suggestions', permission: 'products_view' },
        { path: '/notes', icon: <FaStickyNote />, label: 'Shop Notes', permission: 'products_view' },
        { path: '/sales', icon: <FaChartLine />, label: 'Sales History', permission: 'sales_view' },
        { path: '/sales/deliveries', icon: <FaTruck />, label: 'Deliveries', permission: 'sales_view' },
        { path: '/quotations', icon: <FaFileInvoice />, label: 'Quotations', permission: 'quotations_view' },
        { path: '/purchases', icon: <FaShoppingBag />, label: 'Purchases', permission: 'purchases_view' },
        { path: '/expenses', icon: <FaMoneyBillWave />, label: 'Expenses', permission: 'expenses_view' },
        { path: '/transfers', icon: <FaExchangeAlt />, label: 'Transfers', permission: 'transfers_view' },
        { path: '/returns', icon: <FaUndo />, label: 'Returns', permission: 'returns_view' },
        { path: '/customers', icon: <FaUsers />, label: 'People', permission: 'customers_view' },
        { path: '/reports', icon: <FaChartLine />, label: 'Reports', permission: 'view_reports' },
        { path: '/ai-analytics', icon: <FaRobot />, label: 'AI Demand Forecast', permission: 'view_reports' },
        { path: '/settings', icon: <FaCog />, label: 'Settings', permission: 'manage_settings' },
    ];

    const logoGradient = 'linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%)';

    return (
        <div className="sidebar" style={{
            width: '260px',
            background: 'var(--sidebar-bg)',
            color: 'var(--sidebar-text)',
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid var(--border-color)',
            boxShadow: '4px 0 10px rgba(0,0,0,0.05)',
            zIndex: 50
        }}>
            <div style={{ padding: '24px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                    width: '36px', height: '36px', borderRadius: '10px',
                    background: logoGradient,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: 'white', fontWeight: 'bold', fontSize: '1.2rem',
                    boxShadow: '0 4px 10px rgba(0,0,0,0.2)'
                }}>P</div>
                <div>
                    <h2 style={{ color: '#fff', margin: 0, fontSize: '1.2rem', fontWeight: '700' }}>SAAI POS</h2>
                    <span style={{ fontSize: '0.7rem', color: 'var(--primary-color)', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: '4px' }}>ENTERPRISE</span>
                </div>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 16px' }}>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {menuItems.map((item, index) => {
                        if (item.header) {
                            return (
                                <li key={`header-${index}`} style={{
                                    padding: '16px 16px 8px',
                                    fontSize: '0.75rem',
                                    fontWeight: '700',
                                    color: 'rgba(255,255,255,0.4)',
                                    textTransform: 'uppercase',
                                    letterSpacing: '1.2px'
                                }}>
                                    {item.header}
                                </li>
                            );
                        }
                        if (item.permission && !hasPermission(item.permission)) return null;
                        return (
                            <li key={item.path}>
                                <NavLink
                                    to={item.path}
                                    end={item.exact}
                                    style={({ isActive }) => ({
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '12px',
                                        padding: '12px 16px',
                                        borderRadius: '12px',
                                        color: isActive ? 'white' : 'var(--sidebar-text)',
                                        textDecoration: 'none',
                                        backgroundImage: isActive ? logoGradient : 'none',
                                        backgroundSize: '200% 200%',
                                        boxShadow: isActive ? '0 4px 15px rgba(0,0,0,0.2)' : 'none',
                                        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                        opacity: isActive ? 1 : 0.8,
                                        fontWeight: isActive ? '600' : '500',
                                        transform: isActive ? 'translateX(4px)' : 'none',
                                        position: 'relative'
                                    })}
                                >
                                    <span style={{ fontSize: '1.1rem' }}>{item.icon}</span>
                                    {item.label}
                                    {item.label === 'Sales History' && unreadCount > 0 && (
                                        <span style={{
                                            position: 'absolute',
                                            right: '12px',
                                            background: '#ef4444',
                                            color: 'white',
                                            minWidth: '20px',
                                            height: '20px',
                                            borderRadius: '10px',
                                            fontSize: '0.65rem',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            fontWeight: '800'
                                        }}>
                                            {unreadCount}
                                        </span>
                                    )}
                                </NavLink>
                            </li>
                        );
                    })}
                </ul>
            </div>

            <div style={{ padding: '20px', borderTop: '1px solid rgba(255,255,255,0.05)', background: 'rgba(0,0,0,0.1)' }}>
                <button
                    onClick={toggleTheme}
                    className="btn"
                    style={{
                        background: 'rgba(255,255,255,0.05)', color: 'inherit', width: '100%',
                        justifyContent: 'space-between', marginBottom: '10px', border: '1px solid rgba(255,255,255,0.05)'
                    }}
                >
                    <span>Theme Mode</span>
                    {settings.theme === 'dark' ? <FaSun /> : <FaMoon />}
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
