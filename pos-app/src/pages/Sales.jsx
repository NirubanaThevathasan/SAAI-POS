import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import SalesList from './sales/SalesList';
import ManualSale from './sales/ManualSale';
import { useAuth } from '../context/AuthContext';

const Sales = () => {
    const { hasPermission } = useAuth();
    const [activeTab, setActiveTab] = useState('list');
    const navigate = useNavigate();

    return (
        <div className="sales-page">
            <h2 className="mb-4">Sales Management</h2>

            <div className="tabs" style={{ display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #ddd' }}>
                <button
                    className={`btn ${activeTab === 'list' ? 'btn-primary' : 'btn-light'}`}
                    onClick={() => setActiveTab('list')}
                    style={{ borderRadius: '5px 5px 0 0' }}
                >
                    Sales List
                </button>
                {hasPermission('pos_access') && (
                    <button
                        className="btn btn-warning"
                        onClick={() => navigate('/pos')}
                        style={{ borderRadius: '5px 5px 0 0' }}
                    >
                        POS Sales (Terminal)
                    </button>
                )}
                {hasPermission('sales_add') && (
                    <button
                        className={`btn ${activeTab === 'add' ? 'btn-primary' : 'btn-light'}`}
                        onClick={() => setActiveTab('add')}
                        style={{ borderRadius: '5px 5px 0 0' }}
                    >
                        Add Sale (Manual)
                    </button>
                )}
            </div>

            <div className="tab-content">
                {activeTab === 'list' && <SalesList />}
                {activeTab === 'add' && <ManualSale />}
            </div>
        </div>
    );
};

export default Sales;
