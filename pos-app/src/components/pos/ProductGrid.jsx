import React, { useState, useEffect, useRef } from 'react';
import { FaSearch } from 'react-icons/fa';
import { useSettings } from '../../context/SettingsContext';

const ProductGrid = ({ products, categories, brands, onAddToCart, getStock }) => {
    const { formatPrice } = useSettings();
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('');
    const [selectedBrand, setSelectedBrand] = useState('');
    const searchInputRef = useRef(null);

    // Auto-focus search on mount
    useEffect(() => {
        if (searchInputRef.current) searchInputRef.current.focus();
    }, []);

    // Filter products
    const filteredProducts = products?.filter(p => {
        const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            (p.code && p.code.toLowerCase().includes(searchTerm.toLowerCase()));
        const matchesCategory = selectedCategory ? p.category_id === parseInt(selectedCategory) : true;
        const matchesBrand = selectedBrand ? p.brand_id === parseInt(selectedBrand) : true;

        return matchesSearch && matchesCategory && matchesBrand;
    });

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            // Exact match logic prioritizes code scan
            const exactMatch = products?.find(p => p.code === searchTerm || p.id.toString() === searchTerm);
            if (exactMatch) {
                onAddToCart(exactMatch);
                setSearchTerm('');
            } else if (filteredProducts?.length === 1) {
                onAddToCart(filteredProducts[0]);
                setSearchTerm('');
            }
        }
    };

    return (
        <div>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                <div style={{ position: 'relative', flex: 2 }}>
                    <FaSearch style={{ position: 'absolute', top: '12px', left: '15px', color: '#888' }} />
                    <input
                        ref={searchInputRef}
                        type="text"
                        className="form-control"
                        placeholder="Scan barcode or search..."
                        style={{ paddingLeft: '40px', height: '45px', fontSize: '1.1rem' }}
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                </div>
                <select className="form-control" style={{ flex: 1 }} value={selectedCategory} onChange={e => setSelectedCategory(e.target.value)}>
                    <option value="">All Categories</option>
                    {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <select className="form-control" style={{ flex: 1 }} value={selectedBrand} onChange={e => setSelectedBrand(e.target.value)}>
                    <option value="">All Brands</option>
                    {brands?.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                </select>
            </div>

            <div className="product-grid" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                gap: '20px'
            }}>
                {filteredProducts?.map(product => {
                    const whStock = getStock ? getStock(product.id) : null;
                    const alertQty = product.alert_quantity || 5;
                    const isLowStock = whStock !== null && whStock <= alertQty && whStock > 0;
                    return (
                        <div
                            key={product.id}
                            className="product-card card"
                            onClick={() => onAddToCart(product)}
                            style={{
                                cursor: 'pointer',
                                textAlign: 'center',
                                transition: 'transform 0.1s',
                                opacity: (whStock !== null && whStock <= 0) ? 0.6 : 1,
                                border: isLowStock ? '2px solid #f59e0b' : '1px solid var(--border-color)',
                                position: 'relative'
                            }}
                        >
                            {isLowStock && (
                                <div style={{
                                    position: 'absolute', top: 8, right: 8,
                                    background: '#f59e0b', color: 'white',
                                    fontSize: '0.65rem', fontWeight: 'bold',
                                    padding: '2px 6px', borderRadius: '4px',
                                    zIndex: 2
                                }}>
                                    LOW
                                </div>
                            )}
                            <div className="product-image" style={{ height: '100px', backgroundColor: '#eee', marginBottom: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <span>{product.code || 'No Code'}</span>
                            </div>
                            <h4 style={{
                                margin: '8px 0',
                                fontSize: '0.95rem',
                                fontWeight: '700',
                                lineHeight: '1.2',
                                wordBreak: 'break-word'
                            }}>{product.name}</h4>
                            <p className="price" style={{ color: 'var(--primary-color)', fontWeight: '800', fontSize: '1.1rem', margin: '4px 0 0' }}>{formatPrice(product.price)}</p>
                            <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '2px 0' }}>Cost: {formatPrice(product.cost || 0)}</p>
                            <p className="stock" style={{
                                fontSize: '0.8rem',
                                color: (whStock !== null && whStock <= 0) ? '#e11d48' : '#64748b',
                                fontWeight: '700',
                                marginTop: '5px',
                                padding: '2px 8px',
                                background: (whStock !== null && whStock <= 0) ? 'rgba(225, 29, 72, 0.1)' : 'rgba(100, 116, 139, 0.1)',
                                borderRadius: '4px',
                                display: 'inline-block'
                            }}>
                                Stock: {whStock !== null ? whStock : product.stock_quantity}
                            </p>
                        </div>
                    );
                })}
                {!filteredProducts?.length && <p>No products found.</p>}
            </div>
        </div>
    );
};

export default ProductGrid;
