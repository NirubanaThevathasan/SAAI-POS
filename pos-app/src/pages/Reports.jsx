import React, { useState } from 'react';
import { FaChartLine, FaShoppingCart, FaBox, FaExclamationTriangle, FaMoneyBillWave, FaUsers, FaCashRegister, FaChartPie } from 'react-icons/fa';
import SalesReport from './reports/SalesReport';
import InventoryReport from './reports/InventoryReport';
import ExpenseReport from './reports/ExpenseReport';
import OverviewReport from './reports/OverviewReport';
import ProfitReport from './reports/ProfitReport'; // We'll rename FinancialReport to this
import PurchaseReport from './reports/PurchaseReport';
import AlertsReport from './reports/AlertsReport';
import PeopleReport from './reports/PeopleReport';
import RegisterReport from './reports/RegisterReport';
import PeriodicSalesReport from './reports/PeriodicSalesReport';
import ProductPerformance from './reports/ProductPerformance';

const Reports = () => {
    const [activeTab, setActiveTab] = useState('overview');

    const renderContent = () => {
        switch (activeTab) {
            case 'overview': return <OverviewReport />;
            case 'performance': return <ProductPerformance />;
            case 'sales': return <SalesReport />;
            case 'purchases': return <PurchaseReport />;
            case 'inventory': return <InventoryReport />;
            case 'alerts': return <AlertsReport />;
            case 'profit': return <ProfitReport />;
            case 'people': return <PeopleReport />;
            case 'register': return <RegisterReport />;
            case 'expenses': return <ExpenseReport />;
            case 'periodic': return <PeriodicSalesReport />;
            default: return <OverviewReport />;
        }
    };

    const navItems = [
        { id: 'overview', label: 'Overview', icon: <FaChartPie /> },
        { id: 'performance', label: 'Product Performance', icon: <FaChartLine /> },
        { id: 'profit', label: 'Profit Report', icon: <FaMoneyBillWave /> },
        { id: 'sales', label: 'Sales Reports', icon: <FaChartLine /> },
        { id: 'purchases', label: 'Purchase Reports', icon: <FaShoppingCart /> },
        { id: 'inventory', label: 'Inventory Reports', icon: <FaBox /> },
        { id: 'alerts', label: 'Alerts', icon: <FaExclamationTriangle /> },
        { id: 'people', label: 'People', icon: <FaUsers /> },
        { id: 'register', label: 'Registers', icon: <FaCashRegister /> },
        { id: 'expenses', label: 'Expenses', icon: <FaMoneyBillWave /> },
        { id: 'periodic', label: 'Periodic Summary', icon: <FaChartLine /> },
    ];

    return (
        <div className="reports-page" style={{ display: 'flex', height: 'calc(100vh - 100px)' }}>
            <div className="reports-sidebar" style={{ width: '220px', borderRight: '1px solid #eee', paddingRight: '20px', overflowY: 'auto' }}>
                <h3 style={{ marginBottom: '20px', paddingLeft: '10px' }}>Reports</h3>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                    {navItems.map(item => (
                        <li
                            key={item.id}
                            onClick={() => setActiveTab(item.id)}
                            style={{
                                padding: '12px 15px',
                                cursor: 'pointer',
                                background: activeTab === item.id ? '#e3f2fd' : 'transparent',
                                color: activeTab === item.id ? '#1976d2' : '#333',
                                borderRadius: '8px',
                                marginBottom: '5px',
                                display: 'flex', alignItems: 'center', gap: '10px',
                                fontWeight: activeTab === item.id ? '600' : 'normal'
                            }}
                        >
                            {item.icon} {item.label}
                        </li>
                    ))}
                </ul>
            </div>
            <div className="reports-content" style={{ flex: 1, padding: '20px', overflowY: 'auto', backgroundColor: '#f8f9fa' }}>
                {renderContent()}
            </div>
        </div>
    );
};

export default Reports;
