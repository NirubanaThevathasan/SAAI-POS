import React, { useState } from 'react';
import ProductList from './products/ProductList';
import QuantityAdjustments from './products/QuantityAdjustments';
import StockCounts from './products/StockCounts';

const Products = () => {
    const [activeTab, setActiveTab] = useState('list');

    return (
        <div className="products-container">
            <h2 className="mb-4">Inventory Management</h2>

            <div className="tabs" style={{ display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #ddd' }}>
                <button
                    className={`btn ${activeTab === 'list' ? 'btn-primary' : 'btn-light'}`}
                    onClick={() => setActiveTab('list')}
                    style={{ borderRadius: '5px 5px 0 0' }}
                >
                    Product List
                </button>
                <button
                    className={`btn ${activeTab === 'adjustments' ? 'btn-primary' : 'btn-light'}`}
                    onClick={() => setActiveTab('adjustments')}
                    style={{ borderRadius: '5px 5px 0 0' }}
                >
                    Quantity Adjustments
                </button>
                <button
                    className={`btn ${activeTab === 'counts' ? 'btn-primary' : 'btn-light'}`}
                    onClick={() => setActiveTab('counts')}
                    style={{ borderRadius: '5px 5px 0 0' }}
                >
                    Stock Counts
                </button>
            </div>

            <div className="tab-content">
                {activeTab === 'list' && <ProductList />}
                {activeTab === 'adjustments' && <QuantityAdjustments />}
                {activeTab === 'counts' && <StockCounts />}
            </div>
        </div>
    );
};

export default Products;
