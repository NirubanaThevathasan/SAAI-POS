import React, { useState } from 'react';
import { FaCogs, FaUsers, FaBox, FaMoneyBillWave, FaPrint, FaEnvelope, FaDatabase } from 'react-icons/fa';
import GeneralSettings from './settings/GeneralSettings';
import Categories from './settings/Categories';
import Brands from './settings/Brands';
import TaxRates from './settings/TaxRates';
import Warehouses from './settings/Warehouses';
import ExpenseCategories from './settings/ExpenseCategories';
import SimpleCrud from './settings/SimpleCrud';
import AccessControl from './settings/AccessControl';
import SystemMaintenance from './settings/SystemMaintenance';
import LANMultiComputerLogin from './settings/LANMultiComputerLogin';

const Settings = () => {
    const [activeTab, setActiveTab] = useState('general');

    const renderContent = () => {
        switch (activeTab) {
            case 'general': return <GeneralSettings />;
            case 'categories': return <Categories />;
            case 'brands': return <Brands />;
            case 'units': return <SimpleCrud title="Units" table="units" fields={[{ key: 'code', label: 'Code' }, { key: 'name', label: 'Name' }]} />;
            case 'taxes': return <TaxRates />;
            case 'warehouses': return <Warehouses />;
            case 'expense_categories': return <ExpenseCategories />;
            case 'currencies': return <SimpleCrud title="Currencies" table="currencies" fields={[{ key: 'code', label: 'Code' }, { key: 'name', label: 'Name' }, { key: 'symbol', label: 'Symbol' }, { key: 'exchange_rate', label: 'Exchange Rate' }]} />;
            case 'customer_groups': return <SimpleCrud title="Customer Groups" table="customer_groups" fields={[{ key: 'name', label: 'Group Name' }, { key: 'percentage', label: 'Discount %' }]} />;
            case 'payment_methods': return <SimpleCrud title="Payment Methods" table="payment_methods" fields={[{ key: 'name', label: 'Method Name' }]} />;
            case 'printers': return <SimpleCrud title="Printers" table="printers" fields={[{ key: 'name', label: 'Name' }, { key: 'type', label: 'Type (Network/Windows)' }, { key: 'ip_address', label: 'IP / Port' }]} />;
            case 'email_templates': return <SimpleCrud title="Email Templates" table="email_templates" fields={[{ key: 'name', label: 'Template Name' }, { key: 'subject', label: 'Subject' }]} />;
            case 'access_control': return <AccessControl />;
            case 'lan_login': return <LANMultiComputerLogin />;
            case 'maintenance': return <SystemMaintenance />;
            default: return <GeneralSettings />;
        }
    };

    const menuGroups = [
        {
            title: 'System',
            icon: <FaCogs />,
            items: [
                { id: 'general', label: 'General / POS' },
                { id: 'lan_login', label: 'LAN Multi-Computer Login' },
                { id: 'warehouses', label: 'Warehouses' },
                { id: 'printers', label: 'Printers' },
                { id: 'email_templates', label: 'Email Templates' },
                { id: 'maintenance', label: 'System Maintenance' },
            ]
        },
        {
            title: 'Products & Inventory',
            icon: <FaBox />,
            items: [
                { id: 'categories', label: 'Categories' },
                { id: 'brands', label: 'Brands' },
                { id: 'units', label: 'Units' },
                { id: 'taxes', label: 'Tax Rates' },
            ]
        },
        {
            title: 'Finance & People',
            icon: <FaUsers />,
            items: [
                { id: 'currencies', label: 'Currencies' },
                { id: 'customer_groups', label: 'Customer Groups' },
                { id: 'expense_categories', label: 'Expense Categories' },
                { id: 'payment_methods', label: 'Payment Methods' },
                { id: 'access_control', label: 'Access Control' },
            ]
        }
    ];

    return (
        <div className="settings-page" style={{ display: 'flex', height: 'calc(100vh - 100px)' }}>
            <div className="settings-sidebar" style={{ width: '240px', borderRight: '1px solid #eee', paddingRight: '20px', overflowY: 'auto' }}>
                <h3 style={{ marginBottom: '20px', paddingLeft: '10px' }}>Settings</h3>
                {menuGroups.map((group, idx) => (
                    <div key={idx} style={{ marginBottom: '20px' }}>
                        <h4 style={{ fontSize: '14px', textTransform: 'uppercase', color: '#888', paddingLeft: '10px', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {group.icon} {group.title}
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            {group.items.map(item => (
                                <li
                                    key={item.id}
                                    onClick={() => setActiveTab(item.id)}
                                    style={{
                                        padding: '10px 15px',
                                        cursor: 'pointer',
                                        background: activeTab === item.id ? '#e3f2fd' : 'transparent',
                                        color: activeTab === item.id ? '#1976d2' : '#333',
                                        borderRadius: '6px',
                                        marginBottom: '2px',
                                        fontSize: '14px'
                                    }}
                                >
                                    {item.label}
                                </li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
            <div className="settings-content" style={{ flex: 1, padding: '20px', overflowY: 'auto', backgroundColor: '#fcfcfc' }}>
                {renderContent()}
            </div>
        </div>
    );
};

export default Settings;
