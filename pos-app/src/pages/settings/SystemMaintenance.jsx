import React, { useState, useEffect } from 'react';
import { db, resetDatabase } from '../../db';
import { downloadBackup, importDatabase } from '../../services/BackupService';
import { getLastBackupTime, saveBackupToStorage } from '../../services/AutoBackupService';
import { FaDatabase, FaExclamationTriangle, FaCheckCircle, FaSpinner, FaDownload, FaUpload, FaSync } from 'react-icons/fa';

const SystemMaintenance = () => {
    const [isResetting, setIsResetting] = useState(false);
    const [confirmed, setConfirmed] = useState(false);
    const [message, setMessage] = useState('');
    const [importing, setImporting] = useState(false);
    const [fileError, setFileError] = useState('');
    const [lastBackup, setLastBackup] = useState(null);
    const [backingUp, setBackingUp] = useState(false);
    const [keepStorage, setKeepStorage] = useState(false);

    useEffect(() => {
        const updateBackupTime = () => {
            setLastBackup(getLastBackupTime());
        };
        updateBackupTime();
        const interval = setInterval(updateBackupTime, 30000); // Update every 30 seconds
        return () => clearInterval(interval);
    }, []);

    const handleExport = async () => {
        try {
            const success = await downloadBackup();
            if (success) {
                setMessage('Backup downloaded successfully!');
                // Also update localStorage backup
                await saveBackupToStorage();
            }
        } catch (error) {
            setMessage('Export failed: ' + error.message);
        }
    };

    const handleManualBackup = async () => {
        setBackingUp(true);
        try {
            const success = await saveBackupToStorage();
            if (success) {
                setMessage('Auto-backup saved to device storage!');
                setLastBackup(getLastBackupTime());
            } else {
                setMessage('Backup failed - data may be too large');
            }
        } catch (error) {
            setMessage('Backup failed: ' + error.message);
        } finally {
            setBackingUp(false);
        }
    };

    const handleImport = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Reset error/message
        setFileError('');
        setMessage('');

        if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
            setFileError('Please select a valid JSON backup file.');
            return;
        }

        if (!window.confirm('WARNING: Importing data will OVERWRITE all current data. This cannot be undone. Are you sure?')) {
            event.target.value = ''; // Reset input
            return;
        }

        setImporting(true);
        const reader = new FileReader();

        reader.onload = async (e) => {
            try {
                const json = JSON.parse(e.target.result);
                await importDatabase(json);
                alert('Data restored successfully! The page will now reload.');
                window.location.reload();
            } catch (error) {
                console.error('Import error:', error);
                setFileError('Failed to import data: ' + error.message);
                setImporting(false);
            }
        };

        reader.onerror = () => {
            setFileError('Error reading file.');
            setImporting(false);
        };

        reader.readAsText(file);
    };

    const handleReset = async () => {
        if (!confirmed) {
            alert('Please check the confirmation box first.');
            return;
        }

        let msg = 'CRITICAL: This will permanently delete ALL sales, products, customers, and data.';
        if (keepStorage) {
            msg = 'CRITICAL: This will delete ALL sales, customers, and transactions. Products & Stock will be PRESERVED.';
        }

        const reallySure = window.confirm(msg + ' This cannot be undone. Are you absolutely sure?');
        if (!reallySure) return;

        setIsResetting(true);
        setMessage('Clearing database... Please wait.');

        try {
            await resetDatabase({ keepStorage });
            // Note: resetDatabase calls window.location.reload()
        } catch (error) {
            console.error('Reset failed:', error);
            setMessage('Reset failed. Please try clearing site data manually in browser.');
            setIsResetting(false);
        }
    };

    return (
        <div style={{ maxWidth: '800px' }}>
            <div style={{ marginBottom: '30px', borderBottom: '1px solid #eee', paddingBottom: '15px' }}>
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <FaDatabase style={{ color: 'var(--primary-color)' }} /> System Maintenance
                </h2>
                <p style={{ color: '#666' }}>Manage database tools and system-wide cleanup tasks.</p>
            </div>

            <div className="card" style={{ marginBottom: '20px', border: '2px solid #10b981', background: '#f0fdf4' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#059669' }}>
                    <FaSync color="#10b981" /> Auto-Backup System
                </h3>
                <p style={{ color: '#666', fontSize: '0.9rem', marginBottom: '15px' }}>
                    Your data is automatically backed up to device storage every 5 minutes and before closing the app.
                    This ensures your data is safe even if the browser data is cleared.
                </p>
                {lastBackup ? (
                    <div style={{ background: '#fff', padding: '12px', borderRadius: '8px', marginBottom: '15px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#059669' }}>
                            <FaCheckCircle />
                            <strong>Last Auto-Backup:</strong> {lastBackup.toLocaleString()}
                        </div>
                    </div>
                ) : (
                    <div style={{ background: '#fef3c7', padding: '12px', borderRadius: '8px', marginBottom: '15px', color: '#92400e' }}>
                        <strong>No backup found.</strong> Create a backup now to protect your data.
                    </div>
                )}
                <button
                    className="btn"
                    onClick={handleManualBackup}
                    disabled={backingUp}
                    style={{ background: '#10b981', color: '#fff', marginTop: '10px' }}
                >
                    {backingUp ? (
                        <><FaSpinner className="spin" style={{ marginRight: '8px' }} /> Backing Up...</>
                    ) : (
                        <><FaSync style={{ marginRight: '8px' }} /> Create Backup Now</>
                    )}
                </button>
            </div>

            <div className="card" style={{ marginBottom: '20px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <FaDownload color="var(--primary-color)" /> Download Backup File
                </h3>
                <p style={{ color: '#666', fontSize: '0.9rem' }}>
                    Download a copy of all your data (products, sales, customers, etc.) to your local device as a JSON file.
                    Keep this file safe to restore your data later or transfer to another computer.
                </p>
                <button
                    className="btn"
                    onClick={handleExport}
                    style={{ background: 'var(--primary-color)', color: '#fff', marginTop: '10px' }}
                >
                    <FaDownload style={{ marginRight: '8px' }} /> Export to JSON File
                </button>
            </div>

            <div className="card" style={{ marginBottom: '20px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <FaUpload color="#d97706" /> Restore Data
                </h3>
                <p style={{ color: '#666', fontSize: '0.9rem' }}>
                    Restore your system from a previously saved backup file.
                    <br />
                    <strong style={{ color: '#d97706' }}>Warning: This will replace all current data.</strong>
                </p>

                <div style={{ marginTop: '15px' }}>
                    <label
                        className="btn"
                        style={{
                            background: '#f59e0b',
                            color: '#fff',
                            display: 'inline-flex',
                            alignItems: 'center',
                            cursor: importing ? 'wait' : 'pointer',
                            opacity: importing ? 0.7 : 1
                        }}
                    >
                        {importing ? <FaSpinner className="spin" style={{ marginRight: '8px' }} /> : <FaUpload style={{ marginRight: '8px' }} />}
                        {importing ? 'Restoring...' : 'Select Backup File'}
                        <input
                            type="file"
                            accept=".json"
                            onChange={handleImport}
                            disabled={importing}
                            style={{ display: 'none' }}
                        />
                    </label>
                </div>
                {fileError && <p style={{ color: 'red', marginTop: '10px', fontSize: '0.9rem' }}>{fileError}</p>}
            </div>

            <div className="card" style={{ border: '1px solid #fee2e2', background: '#fef2f2' }}>
                <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
                    <div style={{ background: '#ef4444', color: '#fff', padding: '15px', borderRadius: '12px' }}>
                        <FaExclamationTriangle fontSize="24px" />
                    </div>
                    <div>
                        <h3 style={{ color: '#991b1b', marginTop: 0 }}>Factory Reset</h3>
                        <p style={{ color: '#b91c1c', fontSize: '0.9rem', lineHeight: '1.5' }}>
                            This action will wipe the system completely clean and restore it to its original state.
                            <strong> All products, sales, purchases, customers, and reports will be permanently deleted.</strong>
                        </p>

                        <div style={{ marginTop: '20px', marginBottom: '20px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#7f1d1d', fontWeight: 'bold' }}>
                                <input
                                    type="checkbox"
                                    checked={confirmed}
                                    onChange={(e) => setConfirmed(e.target.checked)}
                                    style={{ width: '18px', height: '18px' }}
                                />
                                I understand that all data will be permanently erased.
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#065f46', fontWeight: 'bold', marginTop: '10px' }}>
                                <input
                                    type="checkbox"
                                    checked={keepStorage}
                                    onChange={(e) => setKeepStorage(e.target.checked)}
                                    style={{ width: '18px', height: '18px' }}
                                />
                                Keep Storage (Products, Inventory, Categories)
                            </label>
                        </div>

                        <button
                            className="btn"
                            onClick={handleReset}
                            disabled={!confirmed || isResetting}
                            style={{
                                background: confirmed ? '#dc2626' : '#f87171',
                                color: '#fff',
                                padding: '12px 24px',
                                fontSize: '1rem',
                                fontWeight: 'bold',
                                opacity: confirmed ? 1 : 0.6
                            }}
                        >
                            {isResetting ? (
                                <><FaSpinner className="spin" /> Resetting System...</>
                            ) : 'Execute Factory Reset'}
                        </button>

                        {message && (
                            <div style={{ marginTop: '15px', color: '#dc2626', fontWeight: 'bold' }}>
                                {message}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="card" style={{ marginTop: '20px' }}>
                <h3>Database Info</h3>
                <p style={{ fontSize: '0.9rem', color: '#666' }}>
                    Database Name: <strong>POSDatabase</strong><br />
                    Storage Engine: <strong>IndexedDB (Dexie)</strong><br />
                    Auto-Backup: <strong style={{ color: '#10b981' }}>Enabled</strong> (localStorage + periodic saves)
                </p>
                <div style={{ background: '#f8fafc', padding: '15px', borderRadius: '8px', fontSize: '0.85rem', marginTop: '10px' }}>
                    <FaCheckCircle style={{ color: '#10b981', marginRight: '8px' }} />
                    <strong>Data Protection:</strong> Your data is automatically backed up every 5 minutes and before closing the app.
                    If IndexedDB is cleared, the app will automatically restore from the backup on next startup.
                </div>
                <div style={{ background: '#fef3c7', padding: '15px', borderRadius: '8px', fontSize: '0.85rem', marginTop: '10px', color: '#92400e' }}>
                    <strong>Important:</strong> For additional safety, regularly download backup files and store them in a safe location.
                    Default configurations (Admin accounts, Currencies, Tax Rates) will be re-seeded automatically after reset.
                </div>
            </div>
        </div>
    );
};

export default SystemMaintenance;
