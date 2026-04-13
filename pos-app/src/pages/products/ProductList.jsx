import React, { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db'; // Adjusted path
import { FaPlus, FaTrash, FaEdit, FaFileImport } from 'react-icons/fa';
import ProductForm from '../ProductForm';
import { useSettings } from '../../context/SettingsContext';
import { useAuth } from '../../context/AuthContext';
import { pricingService } from '../../services/pricingService';

const ProductList = () => {
    const { formatPrice } = useSettings();
    const { hasPermission } = useAuth();
    const products = useLiveQuery(() => db.products.toArray());
    const categories = useLiveQuery(() => db.categories.toArray());
    const brands = useLiveQuery(() => db.brands.toArray());
    const warehouses = useLiveQuery(() => db.warehouses.toArray());
    const stocks = useLiveQuery(() => db.stock.toArray());

    const [showForm, setShowForm] = useState(false);
    const [showImport, setShowImport] = useState(false);
    const [editingProduct, setEditingProduct] = useState(null);

    // AI pricing state per product: { [productId]: { loading, decision, error } }
    const [pricingState, setPricingState] = useState({});

    // Filters
    const [searchTerm, setSearchTerm] = useState('');
    const [filterCategory, setFilterCategory] = useState('');
    const [filterBrand, setFilterBrand] = useState('');
    const [filterWarehouse, setFilterWarehouse] = useState('');

    const filteredProducts = products?.filter(p => {
        const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            p.code.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesCategory = filterCategory ? p.category_id === parseInt(filterCategory) : true;
        const matchesBrand = filterBrand ? p.brand_id === parseInt(filterBrand) : true;
        return matchesSearch && matchesCategory && matchesBrand;
    });

    const handleDelete = async (id) => {
        if (window.confirm('Delete product?')) await db.products.delete(id);
    };

    const handleEdit = (product) => {
        setEditingProduct(product);
        setShowForm(true);
    };

    const handleFormClose = () => {
        setShowForm(false);
        setEditingProduct(null);
    };

    const handleAiSuggest = async (product) => {
        if (!pricingService) return;

        setPricingState(prev => ({
            ...prev,
            [product.id]: {
                ...(prev[product.id] || {}),
                loading: true,
                error: null
            }
        }));

        try {
            const decision = await pricingService.getSuggestionForProduct(product.id);
            setPricingState(prev => ({
                ...prev,
                [product.id]: {
                    ...(prev[product.id] || {}),
                    loading: false,
                    decision,
                    error: null
                }
            }));
        } catch (error) {
            console.error('AI pricing error', error);
            setPricingState(prev => ({
                ...prev,
                [product.id]: {
                    ...(prev[product.id] || {}),
                    loading: false,
                    error: error.message || 'Failed to get suggestion'
                }
            }));
        }
    };

    const handleApplySuggestion = async (product) => {
        const state = pricingState[product.id];
        const decision = state?.decision;
        if (!decision) return;

        const confirm = window.confirm(
            `Apply AI suggested price ${formatPrice(decision.suggested_price)} for "${product.name}"?`
        );
        if (!confirm) return;

        await db.products.update(product.id, { price: decision.suggested_price });
        setPricingState(prev => {
            const newState = { ...prev };
            delete newState[product.id];
            return newState;
        });
        alert('AI suggested price applied successfully.');
    };

    const handleImport = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = async (event) => {
            const csv = event.target.result;
            const lines = csv.split('\n');
            const productsToAdd = [];
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                const [name, code, category, cost, price] = line.split(',');
                if (name && code && price) {
                    productsToAdd.push({
                        name: name.trim(),
                        code: code.trim(),
                        category_id: category || 'General',
                        brand_id: '',
                        cost: parseFloat(cost) || 0,
                        price: parseFloat(price) || 0,
                        stock_quantity: 0,
                        alert_quantity: 5,
                        tax_method: 'exclusive',
                        type: 'standard'
                    });
                }
            }
            if (productsToAdd.length > 0) {
                await db.products.bulkAdd(productsToAdd);
                alert(`Imported ${productsToAdd.length} products!`);
                setShowImport(false);
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="products-page">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Products</h2>
                <div>
                    {hasPermission('products_add') && (
                        <button className="btn" onClick={() => setShowImport(!showImport)} style={{ marginRight: '10px' }}>
                            <FaFileImport /> Import CSV
                        </button>
                    )}
                    {hasPermission('products_add') && (
                        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
                            <FaPlus /> Add Product
                        </button>
                    )}
                </div>
            </div>

            {showImport && (
                <div className="card mb-3" style={{ border: '1px solid #3498db' }}>
                    <h3>Import Products (CSV)</h3>
                    <p>Format: Name, Code, Category, Cost, Price</p>
                    <input type="file" accept=".csv" onChange={handleImport} />
                </div>
            )}

            {showForm ? (
                <ProductForm onClose={handleFormClose} initialData={editingProduct} />
            ) : (
                <div className="card">
                    <div className="filters mb-3" style={{ display: 'flex', gap: '10px' }}>
                        <input
                            placeholder="Search Name or Code..."
                            className="form-control"
                            style={{ flex: 2 }}
                            value={searchTerm}
                            onChange={e => setSearchTerm(e.target.value)}
                        />
                        <select className="form-control" style={{ flex: 1 }} value={filterCategory} onChange={e => setFilterCategory(e.target.value)}>
                            <option value="">All Categories</option>
                            {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                        </select>
                        <select className="form-control" style={{ flex: 1 }} value={filterBrand} onChange={e => setFilterBrand(e.target.value)}>
                            <option value="">All Brands</option>
                            {brands?.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                        </select>
                        <select className="form-control" style={{ flex: 1 }} value={filterWarehouse} onChange={e => setFilterWarehouse(e.target.value)}>
                            <option value="">All Warehouses (Global Stock)</option>
                            {warehouses?.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                        </select>
                    </div>

                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                                <th style={{ padding: '10px' }}>Name</th>
                                <th style={{ padding: '10px' }}>Code</th>
                                <th style={{ padding: '10px' }}>Type</th>
                                <th style={{ padding: '10px' }}>Price</th>
                                <th style={{ padding: '10px' }}>Stock</th>
                                <th style={{ padding: '10px' }}>AI Pricing</th>
                                <th style={{ padding: '10px' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredProducts?.map(product => (
                                <tr key={product.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                    <td style={{ padding: '10px' }}>{product.name}</td>
                                    <td style={{ padding: '10px' }}>{product.code}</td>
                                    <td style={{ padding: '10px' }}>{product.type || 'Standard'}</td>
                                    <td style={{ padding: '10px' }}>{formatPrice(product.price)}</td>
                                    <td style={{ padding: '10px' }}>
                                        {filterWarehouse ? (
                                            (() => {
                                                const s = stocks?.find(st => st.product_id === product.id && st.warehouse_id === parseInt(filterWarehouse));
                                                return <span style={{ fontWeight: 'bold', color: 'var(--primary-color)' }}>{s ? s.quantity : 0}</span>;
                                            })()
                                        ) : (
                                            product.stock_quantity || 0
                                        )}
                                    </td>
                                    <td style={{ padding: '10px', minWidth: '220px' }}>
                                        {pricingState[product.id]?.decision ? (
                                            <div style={{ fontSize: '0.8rem', marginBottom: '6px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
                                                    <strong style={{ color: 'var(--primary-color)', fontSize: '0.95rem' }}>
                                                        {formatPrice(pricingState[product.id].decision.suggested_price)}
                                                    </strong>
                                                    {pricingState[product.id].decision.suggested_discount_pct > 0 && (
                                                        <span style={{ background: '#fee2e2', color: '#b91c1c', borderRadius: '4px', padding: '1px 5px', fontSize: '0.72rem', fontWeight: 'bold' }}>
                                                            -{pricingState[product.id].decision.suggested_discount_pct}%
                                                        </span>
                                                    )}
                                                </div>
                                                <div style={{ color: '#475569', fontSize: '0.75rem', marginBottom: '3px', fontStyle: 'italic' }}>
                                                    {pricingState[product.id].decision.reason}
                                                </div>
                                                <div style={{ color: '#94a3b8', fontSize: '0.68rem' }}>
                                                    🤖 {pricingState[product.id].decision.calculation_method}
                                                </div>
                                            </div>
                                        ) : pricingState[product.id]?.error ? (
                                            <div style={{ fontSize: '0.75rem', color: '#dc2626', marginBottom: '4px' }}>
                                                ⚠️ {pricingState[product.id].error}
                                            </div>
                                        ) : null}
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                            <button
                                                type="button"
                                                className="btn btn-sm"
                                                onClick={() => handleAiSuggest(product)}
                                                disabled={pricingState[product.id]?.loading}
                                                style={{ padding: '4px 8px' }}
                                            >
                                                {pricingState[product.id]?.loading ? 'Thinking…' : 'Suggest'}
                                            </button>
                                            {pricingState[product.id]?.decision && hasPermission('products_edit') && (
                                                <button
                                                    type="button"
                                                    className="btn btn-sm btn-primary"
                                                    onClick={() => handleApplySuggestion(product)}
                                                    style={{ padding: '4px 8px' }}
                                                >
                                                    Apply
                                                </button>
                                            )}
                                        </div>
                                    </td>
                                    <td style={{ padding: '10px', display: 'flex', gap: '10px' }}>
                                        {hasPermission('products_edit') && (
                                            <button className="btn-icon" style={{ color: 'blue' }} onClick={() => handleEdit(product)} title="Edit">
                                                <FaEdit />
                                            </button>
                                        )}
                                        {hasPermission('products_delete') && (
                                            <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(product.id)} title="Delete">
                                                <FaTrash />
                                            </button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {filteredProducts?.length === 0 && <tr><td colSpan="6" style={{ textAlign: 'center', padding: '20px' }}>No products found</td></tr>}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default ProductList;
