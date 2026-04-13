import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FaHome, FaBox, FaShoppingCart, FaChartLine, FaUsers, FaCog, FaSignOutAlt, FaTruck, FaExchangeAlt, FaUndo, FaMoneyBillWave, FaShoppingBag, FaMoon, FaSun, FaFileInvoice } from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';
import { useSettings } from '../context/SettingsContext';

const Navbar = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const { settings, toggleTheme } = useSettings();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const hasPermission = (perm) => {
        if (!user) return false;
        if (user.role === 'admin') return true;
        return user.permissions?.includes(perm);
    };

    const menuItems = [
        { path: '/', icon: <FaHome />, label: 'Dashboard', exact: true },
        { path: '/pos', icon: <FaShoppingCart />, label: 'POS', permission: 'pos_access' },
        { path: '/products', icon: <FaBox />, label: 'Products', permission: 'products_view' },
        { path: '/sales', icon: <FaChartLine />, label: 'Sales', permission: 'sales_view' },
        { path: '/sales/deliveries', icon: <FaTruck />, label: 'Deliveries', permission: 'sales_view' },
        { path: '/quotations', icon: <FaFileInvoice />, label: 'Quotations', permission: 'quotations_view' },
        { path: '/purchases', icon: <FaShoppingBag />, label: 'Purchases', permission: 'purchases_view' },
        { path: '/expenses', icon: <FaMoneyBillWave />, label: 'Expenses', permission: 'expenses_view' },
        { path: '/people', icon: <FaUsers />, label: 'People', permission: 'customers_view' }, // Condensed People
        { path: '/reports', icon: <FaChartLine />, label: 'Reports', permission: 'view_reports' },
        { path: '/settings', icon: <FaCog />, label: 'Settings', permission: 'manage_settings' },
    ];

    return (
        <div className="navbar" style={{
            height: '64px',
            background: 'var(--sidebar-bg)',
            color: 'var(--sidebar-text)',
            display: 'flex',
            alignItems: 'center',
            padding: '0 24px',
            borderBottom: '1px solid var(--border-color)',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            zIndex: 100
        }}>
            {/* Logo */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: '200px' }}>
                <div style={{ width: '32px', height: '32px', background: 'var(--primary-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 'bold' }}>P</div>
                <h2 style={{ fontSize: '1.2rem', margin: 0, color: 'var(--primary-color)' }}>POS</h2>
            </div>

            {/* Menu */}
            <div style={{ flex: 1, overflowX: 'auto', display: 'flex', alignItems: 'center', gap: '4px', padding: '0 20px' }}>
                {menuItems.map(item => {
                    if (item.permission && !hasPermission(item.permission)) return null;
                    return (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            end={item.exact}
                            style={({ isActive }) => ({
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '8px 12px',
                                borderRadius: '6px',
                                color: isActive ? 'white' : 'inherit',
                                textDecoration: 'none',
                                backgroundColor: isActive ? 'var(--primary-color)' : 'transparent',
                                transition: 'all 0.2s',
                                fontSize: '0.9rem',
                                whiteSpace: 'nowrap'
                            })}
                        >
                            {item.icon}
                            {item.label}
                        </NavLink>
                    );
                })}
            </div>

            {/* Right Side */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', minWidth: 'max-content' }}>
                <button onClick={toggleTheme} className="btn" style={{ background: 'rgba(255,255,255,0.1)', color: 'inherit', padding: '8px', borderRadius: '50%' }}>
                    {settings.theme === 'dark' ? <FaSun /> : <FaMoon />}
                </button>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '16px' }}>
                    <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--primary-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '0.8rem' }}>
                        {user?.username?.[0]?.toUpperCase()}
                    </div>
                    <button onClick={handleLogout} style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', opacity: 0.7 }}>
                        <FaSignOutAlt />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Navbar;
