import React, { useEffect, useState } from 'react';
import { db } from '../../db';

const GeneralSettings = () => {
    const [settings, setSettings] = useState({
        site_name: '',
        company_phone: '',
        company_email: '',
        company_address: '',
        logo_url: '',
        currency_code: 'USD',
        date_format: 'YYYY-MM-DD',
        time_format: '24h',
        default_tax_rate: '',
        default_warehouse: '',
        pos_receipt_header: '',
        pos_receipt_footer: 'Thank you for shopping with us!',
    });

    useEffect(() => {
        const loadSettings = async () => {
            const all = await db.settings.toArray();
            const settingsMap = {};
            all.forEach(s => settingsMap[s.key] = s.value);
            setSettings(prev => ({ ...prev, ...settingsMap }));
        };
        loadSettings();
    }, []);

    const handleChange = (key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    };

    const handleSave = async () => {
        const pairs = Object.entries(settings).map(([key, value]) => ({ key, value }));
        await db.settings.bulkPut(pairs);
        alert('Settings saved successfully!');
    };

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3>General & POS Settings</h3>
                <button className="btn btn-primary" onClick={handleSave}>Update Settings</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* Company Information */}
                <div className="card">
                    <h4 style={{ marginBottom: '15px', color: '#555', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>Company Information</h4>
                    <div className="form-group">
                        <label>Site Name</label>
                        <input type="text" className="form-control" value={settings.site_name} onChange={e => handleChange('site_name', e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label>Company Phone</label>
                        <input type="text" className="form-control" value={settings.company_phone} onChange={e => handleChange('company_phone', e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label>Company Email</label>
                        <input type="email" className="form-control" value={settings.company_email} onChange={e => handleChange('company_email', e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label>Address</label>
                        <textarea className="form-control" rows="2" value={settings.company_address} onChange={e => handleChange('company_address', e.target.value)} />
                    </div>
                </div>

                {/* Locale & Defaults */}
                <div className="card">
                    <h4 style={{ marginBottom: '15px', color: '#555', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>Locale & Defaults</h4>
                    <div className="form-group">
                        <label>Currency Code</label>
                        <input type="text" className="form-control" value={settings.currency_code} onChange={e => handleChange('currency_code', e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label>Date Format</label>
                        <select className="form-control" value={settings.date_format} onChange={e => handleChange('date_format', e.target.value)}>
                            <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                            <option value="DD-MM-YYYY">DD-MM-YYYY</option>
                            <option value="MM/DD/YYYY">MM/DD/YYYY</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Default Tax Rate (%)</label>
                        <input type="number" className="form-control" value={settings.default_tax_rate} onChange={e => handleChange('default_tax_rate', e.target.value)} />
                    </div>
                </div>

                {/* POS Configurations */}
                <div className="card" style={{ gridColumn: '1 / -1' }}>
                    <h4 style={{ marginBottom: '15px', color: '#555', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>POS Configuration (Receipts)</h4>
                    <div className="form-group">
                        <label>Receipt Header (Shop Name/Slogan)</label>
                        <input type="text" className="form-control" value={settings.pos_receipt_header} onChange={e => handleChange('pos_receipt_header', e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label>Receipt Footer (Thank you note)</label>
                        <textarea className="form-control" rows="2" value={settings.pos_receipt_footer} onChange={e => handleChange('pos_receipt_footer', e.target.value)} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GeneralSettings;
