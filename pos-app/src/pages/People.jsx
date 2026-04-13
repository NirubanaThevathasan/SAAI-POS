import React, { useState } from 'react';
import Users from './people/Users';
import Customers from './people/Customers';
import Suppliers from './people/Suppliers';
import Billers from './people/Billers';
import { useAuth } from '../context/AuthContext';

const People = () => {
    const { hasPermission } = useAuth();
    const [activeTab, setActiveTab] = useState('customers');

    const renderContent = () => {
        switch (activeTab) {
            case 'users': return <Users />;
            case 'customers': return <Customers />;
            case 'suppliers': return <Suppliers />;
            case 'billers': return <Billers />;
            default: return <Customers />;
        }
    };

    return (
        <div className="people-page" style={{ display: 'flex', height: 'calc(100vh - 100px)' }}>
            <div className="people-sidebar" style={{ width: '200px', borderRight: '1px solid #eee', paddingRight: '20px' }}>
                <h3 style={{ marginBottom: '20px' }}>People</h3>
                <ul style={{ listStyle: 'none' }}>
                    {hasPermission('customers_view') && (
                        <li
                            onClick={() => setActiveTab('customers')}
                            style={{ padding: '10px', cursor: 'pointer', background: activeTab === 'customers' ? '#e1f5fe' : 'transparent', borderRadius: '4px' }}
                        >
                            Customers
                        </li>
                    )}
                    {hasPermission('suppliers_view') && (
                        <li
                            onClick={() => setActiveTab('suppliers')}
                            style={{ padding: '10px', cursor: 'pointer', background: activeTab === 'suppliers' ? '#e1f5fe' : 'transparent', borderRadius: '4px' }}
                        >
                            Suppliers
                        </li>
                    )}
                    <li
                        onClick={() => setActiveTab('billers')}
                        style={{ padding: '10px', cursor: 'pointer', background: activeTab === 'billers' ? '#e1f5fe' : 'transparent', borderRadius: '4px' }}
                    >
                        Billers
                    </li>
                    {hasPermission('users_view') && (
                        <li
                            onClick={() => setActiveTab('users')}
                            style={{ padding: '10px', cursor: 'pointer', background: activeTab === 'users' ? '#e1f5fe' : 'transparent', borderRadius: '4px' }}
                        >
                            Users (Staff)
                        </li>
                    )}
                </ul>
            </div>
            <div className="people-content" style={{ flex: 1, padding: '0 20px', overflowY: 'auto' }}>
                {renderContent()}
            </div>
        </div>
    );
};

export default People;
