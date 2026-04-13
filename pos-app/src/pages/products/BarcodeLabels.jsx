import React, { useState, useRef } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db } from '../../db';
import Barcode from 'react-barcode';
import { useReactToPrint } from 'react-to-print';

import { useSettings } from '../../context/SettingsContext';

const BarcodeLabels = () => {
    const { formatPrice } = useSettings();
    const products = useLiveQuery(() => db.products.toArray());
    const [selectedProducts, setSelectedProducts] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');

    // Print ref
    const componentRef = useRef();
    const handlePrint = useReactToPrint({
        contentRef: componentRef,
        documentTitle: 'Barcode Labels',
    });

    const addToQueue = (product) => {
        const existing = selectedProducts.find(p => p.id === product.id);
        if (existing) {
            setSelectedProducts(selectedProducts.map(p => p.id === product.id ? { ...p, count: p.count + 1 } : p));
        } else {
            setSelectedProducts([...selectedProducts, { ...product, count: 1 }]);
        }
    };

    const updateCount = (id, count) => {
        if (count < 1) {
            setSelectedProducts(selectedProducts.filter(p => p.id !== id));
        } else {
            setSelectedProducts(selectedProducts.map(p => p.id === id ? { ...p, count } : p));
        }
    };

    const filteredProducts = products?.filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()));

    return (
        <div className="barcode-module">
            <div className="no-print">
                <h2 style={{ marginBottom: '24px', fontWeight: '800' }}>Barcode Label Printer</h2>
                <div style={{ display: 'flex', gap: '20px', height: '80vh' }}>

                    {/* Left: Product Selector */}
                    <div className="card" style={{ flex: 1, overflowY: 'auto' }}>
                        <input
                            placeholder="Search Products..."
                            className="form-control mb-3"
                            value={searchTerm}
                            onChange={e => setSearchTerm(e.target.value)}
                        />
                        <div className="list-group">
                            {filteredProducts?.map(p => (
                                <div key={p.id} className="list-group-item" style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', borderBottom: '1px solid #eee' }}>
                                    <div>
                                        <strong>{p.name}</strong><br />
                                        <small>{p.code}</small>
                                    </div>
                                    <button className="btn btn-sm btn-primary" onClick={() => addToQueue(p)}>Add</button>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right: Print Queue & Preview */}
                    <div className="card" style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
                        <div style={{ marginBottom: '20px', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                            <h4>Print Queue</h4>
                            {selectedProducts.map(p => (
                                <div key={p.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                                    <span>{p.name}</span>
                                    <input type="number" value={p.count} onChange={e => updateCount(p.id, parseInt(e.target.value))} style={{ width: '60px' }} />
                                </div>
                            ))}
                            {selectedProducts.length === 0 && <p style={{ color: '#888' }}>No items in print queue.</p>}

                            <button className="btn btn-success mt-3" onClick={handlePrint} disabled={selectedProducts.length === 0}>Print Labels</button>
                            <button className="btn btn-secondary mt-3 ms-2" onClick={() => setSelectedProducts([])}>Clear</button>
                        </div>

                        {/* Preview Area (Hidden by default in print until targeted) */}
                        <div className="preview-area" style={{ flex: 1, background: '#f8fafc', padding: '20px', overflowY: 'auto', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                            <div ref={componentRef} id="print-section" className="print-section" style={{
                                background: 'white',
                                padding: '15px',
                                display: 'grid',
                                gridTemplateColumns: 'repeat(3, 1fr)',
                                gap: '15px',
                                minHeight: '100%'
                            }}>
                                {selectedProducts.flatMap(p => Array(p.count).fill(p)).map((p, i) => (
                                    <div key={i} className="label-item" style={{
                                        border: '1px solid #e2e8f0',
                                        padding: '12px',
                                        textAlign: 'center',
                                        breakInside: 'avoid',
                                        borderRadius: '8px',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                    }}>
                                        <div style={{ fontSize: '11px', fontWeight: '700', color: '#1e293b', marginBottom: '4px' }}>{p.name}</div>
                                        <div style={{ margin: '4px 0' }}>
                                            <Barcode value={p.code || p.id} width={1.2} height={40} fontSize={10} background="transparent" />
                                        </div>
                                        <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--primary-color)' }}>{formatPrice(p.price)}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Print Specific Styles */}
            <style>{`
                @media print {
                    @page {
                        size: auto;
                        margin: 10mm;
                    }
                    body {
                        background: white !important;
                        margin: 0;
                        padding: 0;
                    }
                    .no-print {
                        display: none !important;
                    }
                    .print-section {
                        display: grid !important;
                        grid-template-columns: repeat(3, 1fr) !important;
                        gap: 10mm !important;
                        padding: 0 !important;
                        margin: 0 !important;
                        visibility: visible !important;
                        width: 100% !important;
                    }
                    .label-item {
                        border: 1px solid #eee !important;
                        padding: 5mm !important;
                        page-break-inside: avoid;
                    }
                }
            `}</style>
        </div>
    );
};

export default BarcodeLabels;
