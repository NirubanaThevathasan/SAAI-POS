import React, { useState, useEffect } from 'react';
import { FaNetworkWired, FaDesktop, FaInfoCircle, FaCopy, FaCheck, FaDatabase, FaUpload, FaDownload } from 'react-icons/fa';
import { publishMainPcSnapshot, syncFromMainPcBackup, isLanShareEnabled, setLanShareEnabled } from '../../services/LanShareService';

// Get local LAN IP via WebRTC (works in browser when app is served with --host)
function getLanIpViaWebRTC() {
    return new Promise((resolve) => {
        const pc = new RTCPeerConnection({ iceServers: [] });
        const noop = () => {};
        pc.createDataChannel('');
        pc.createOffer().then(offer => pc.setLocalDescription(offer)).catch(() => resolve(null));
        pc.onicecandidate = (ice) => {
            if (!ice || !ice.candidate || !ice.candidate.candidate) return;
            const match = /([0-9]{1,3}\.){3}[0-9]{1,3}/.exec(ice.candidate.candidate);
            if (match && match[0] !== '127.0.0.1') {
                resolve(match[0]);
            }
        };
        setTimeout(() => {
            pc.close();
            resolve(null);
        }, 3000);
    });
}

const LANMultiComputerLogin = () => {
    const [lanUrl, setLanUrl] = useState('');
    const [allIPs, setAllIPs] = useState([]);
    const [copied, setCopied] = useState(false);
    const [detecting, setDetecting] = useState(true);
    const [lanShareEnabled, setLanShareEnabledState] = useState(false);
    const [publishMsg, setPublishMsg] = useState('');
    const [syncMsg, setSyncMsg] = useState('');
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        let cancelled = false;
        const { protocol, port } = window.location;
        const portPart = port ? `:${port}` : ':3000';

        const applyUrl = (lanIp, ips = []) => {
            if (cancelled) return;
            const host = lanIp || 'localhost';
            setLanUrl(`${protocol}//${host}${portPart}`);
            setAllIPs(ips);
            setDetecting(false);
        };

        (async () => {
            // Ensure Electron API has getLanAddress if not already set
            if (typeof window !== 'undefined' && typeof require !== 'undefined') {
                try {
                    const { ipcRenderer } = require('electron');
                    window.electronAPI = window.electronAPI || {};
                    if (!window.electronAPI.getLanAddress) {
                        window.electronAPI.getLanAddress = () => ipcRenderer.invoke('lan:get-address');
                    }
                } catch (e) { /* not in Electron */ }
            }

            // 1) In Electron: get LAN IP from main process
            if (typeof window !== 'undefined' && window.electronAPI?.getLanAddress) {
                try {
                    const res = await window.electronAPI.getLanAddress();
                    if (res?.lanIp) {
                        applyUrl(res.lanIp, res.allIPs || []);
                        return;
                    }
                } catch (e) {
                    console.warn('getLanAddress failed', e);
                }
            }

            // 2) Browser / web: if already opened via LAN IP, use it directly
            const currentHost = window.location.hostname;
            if (currentHost && currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
                applyUrl(currentHost, [{ ip: currentHost, name: 'Current Host' }]);
                return;
            }

            // 3) Browser (localhost): try WebRTC to discover *this device* LAN IP (useful on the main PC browser)
            const webrtcIp = await getLanIpViaWebRTC();
            applyUrl(webrtcIp || null, webrtcIp ? [{ ip: webrtcIp, name: 'Detected' }] : []);
        })();

        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        // Load whether main-PC LAN sharing is enabled (only meaningful on the main PC)
        isLanShareEnabled().then(setLanShareEnabledState).catch(() => {});
    }, []);

    const handleCopy = () => {
        if (!lanUrl) return;
        navigator.clipboard.writeText(lanUrl).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    const isElectron = typeof window !== 'undefined' && typeof require !== 'undefined';

    const backupUrl = (() => {
        try {
            const u = new URL(lanUrl);
            return `${u.protocol}//${u.hostname}:4310/backup.json`;
        } catch {
            return '';
        }
    })();

    const handleToggleLanShare = async (enabled) => {
        setLanShareEnabledState(enabled);
        await setLanShareEnabled(enabled);
        setPublishMsg(enabled ? 'LAN data sharing enabled. Publishing snapshots every 60 seconds.' : 'LAN data sharing disabled.');
        if (enabled) {
            try {
                await publishMainPcSnapshot();
                setPublishMsg('Published snapshot. Other devices can now sync.');
            } catch (e) {
                setPublishMsg(`Publish failed: ${e.message}`);
            }
        }
    };

    const handlePublishNow = async () => {
        setBusy(true);
        setPublishMsg('');
        try {
            const res = await publishMainPcSnapshot();
            if (res.status === 'success') {
                setPublishMsg(`Published snapshot (${Math.round((res.bytes || 0) / 1024)} KB).`);
            } else {
                setPublishMsg(res.message || 'Publish failed.');
            }
        } catch (e) {
            setPublishMsg(`Publish failed: ${e.message}`);
        } finally {
            setBusy(false);
        }
    };

    const handleSyncFromMain = async () => {
        if (!backupUrl) {
            setSyncMsg('Backup URL not ready yet.');
            return;
        }
        setBusy(true);
        setSyncMsg('');
        try {
            await syncFromMainPcBackup(backupUrl);
            setSyncMsg('Sync complete. Reloading...');
            setTimeout(() => window.location.reload(), 500);
        } catch (e) {
            setSyncMsg(`Sync failed: ${e.message}`);
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="lan-multi-login">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                <FaNetworkWired style={{ fontSize: '28px', color: '#4f46e5' }} />
                <div>
                    <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                        LAN Multi-Computer Login
                    </h3>
                    <p style={{ margin: '4px 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Allow other computers on your local network to log in to this POS
                    </p>
                </div>
            </div>

            <div className="card" style={{ marginBottom: '24px', borderLeft: '4px solid #4f46e5' }}>
                <h4 style={{ margin: '0 0 12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FaDesktop /> Login URL for other computers
                </h4>
                <p style={{ margin: '0 0 16px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Share this address with other PCs on your network. They open it in a browser and sign in with their user account.
                </p>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '12px' }}>
                    <code style={{
                        flex: '1',
                        minWidth: '200px',
                        padding: '12px 16px',
                        background: 'var(--bg-color)',
                        borderRadius: '8px',
                        border: '1px solid #e2e8f0',
                        fontSize: '0.95rem',
                        fontFamily: 'monospace'
                    }}>
                        {detecting ? 'Detecting LAN address...' : (lanUrl || 'Unable to detect')}
                    </code>
                    <button
                        type="button"
                        onClick={handleCopy}
                        disabled={!lanUrl || detecting}
                        className="btn"
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            background: copied ? '#10b981' : '#4f46e5',
                            color: 'white',
                            border: 'none',
                            padding: '10px 18px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontWeight: '600'
                        }}
                    >
                        {copied ? <><FaCheck /> Copied</> : <><FaCopy /> Copy URL</>}
                    </button>
                </div>
                {allIPs.length > 1 && !detecting && (
                    <div style={{ marginTop: '12px', padding: '12px', background: '#f8fafc', borderRadius: '8px', fontSize: '0.85rem' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>All available IPs:</strong> If the URL above doesn't work, try these:
                        <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {allIPs.map((item, idx) => {
                                const url = `${window.location.protocol}//${item.ip}${window.location.port ? `:${window.location.port}` : ':3000'}`;
                                return (
                                    <code key={idx} style={{
                                        padding: '6px 10px',
                                        background: 'white',
                                        borderRadius: '6px',
                                        border: '1px solid #e2e8f0',
                                        fontSize: '0.8rem',
                                        cursor: 'pointer',
                                        display: 'inline-block'
                                    }} onClick={() => {
                                        setLanUrl(url);
                                        navigator.clipboard.writeText(url);
                                        setCopied(true);
                                        setTimeout(() => setCopied(false), 2000);
                                    }} title={`Click to use: ${item.name}`}>
                                        {item.ip} <span style={{ color: '#64748b', fontSize: '0.75rem' }}>({item.name})</span>
                                    </code>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            <div className="card" style={{ marginBottom: '24px', borderLeft: '4px solid #0ea5e9' }}>
                <h4 style={{ margin: '0 0 12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FaDatabase /> LAN Data Sync (view main PC data)
                </h4>
                <p style={{ margin: '0 0 14px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    This app stores data locally per device. To view the <strong>main PC data</strong> on other devices, sync a snapshot over LAN.
                </p>

                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '12px' }}>
                    <code style={{
                        flex: 1,
                        minWidth: '200px',
                        padding: '10px 14px',
                        background: 'var(--bg-color)',
                        borderRadius: '8px',
                        border: '1px solid #e2e8f0',
                        fontSize: '0.9rem',
                        fontFamily: 'monospace'
                    }}>
                        {backupUrl || 'Backup URL will appear after IP detection'}
                    </code>
                </div>

                {isElectron && (
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '12px' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                            <input
                                type="checkbox"
                                checked={lanShareEnabled}
                                onChange={(e) => handleToggleLanShare(e.target.checked)}
                            />
                            <span>Share Main PC Data to LAN (auto publish)</span>
                        </label>
                        <button
                            type="button"
                            className="btn"
                            onClick={handlePublishNow}
                            disabled={busy}
                            style={{ background: '#0ea5e9', color: 'white', border: 'none', borderRadius: '8px', padding: '10px 16px', fontWeight: 700 }}
                        >
                            <FaUpload style={{ marginRight: '8px' }} />
                            Publish Now
                        </button>
                    </div>
                )}

                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <button
                        type="button"
                        className="btn"
                        onClick={handleSyncFromMain}
                        disabled={busy || !backupUrl}
                        style={{ background: '#10b981', color: 'white', border: 'none', borderRadius: '8px', padding: '10px 16px', fontWeight: 700 }}
                    >
                        <FaDownload style={{ marginRight: '8px' }} />
                        Sync From Main PC
                    </button>
                    {(publishMsg || syncMsg) && (
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            {publishMsg || syncMsg}
                        </div>
                    )}
                </div>

                <div style={{ marginTop: '10px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Snapshot sync is for viewing the main PC data. For true multi-user live updates, you’ll need a shared server database/API.
                </div>
            </div>

            <div className="card" style={{ marginBottom: '24px', background: '#f0f9ff', border: '1px solid #bae6fd' }}>
                <h4 style={{ margin: '0 0 10px', fontWeight: '600', color: '#0369a1', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FaInfoCircle /> How to enable LAN access
                </h4>
                <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.8', color: '#0c4a6e', fontSize: '0.9rem' }}>
                    <li>The URL above is detected automatically. Use it on other devices (phones, tablets, other PCs) on the same Wi‑Fi or LAN.</li>
                    <li>For <strong>web</strong> (browser) access from other devices, run the dev server with <code style={{ background: '#e0f2fe', padding: '2px 6px', borderRadius: '4px' }}>npm run dev -- --host</code> so it listens on the network.</li>
                    <li>In <strong>Electron</strong>, the URL uses this PC’s LAN IP so other devices can open it in a browser (they must connect to the same port where the app is served, e.g. 3000).</li>
                    <li>On other computers, open the URL in Chrome or Edge and log in with a user from <strong>Settings → Access Control</strong>.</li>
                </ul>
            </div>

            <div className="card" style={{ marginBottom: '24px', background: '#fff7ed', border: '1px solid #fed7aa' }}>
                <h4 style={{ margin: '0 0 10px', fontWeight: '600', color: '#c2410c', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FaInfoCircle /> Troubleshooting: Can't connect from phone?
                </h4>
                <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.8', color: '#9a3412', fontSize: '0.9rem' }}>
                    <li><strong>Windows Firewall:</strong> Windows may block port 3000. Open Windows Defender Firewall → Allow an app → Find "Node.js" or "Vite" → Check both Private and Public, or add a rule for port 3000.</li>
                    <li><strong>Try different IP:</strong> If the main URL doesn't work, try the other IPs shown above (click to copy). Your phone might be on a different network segment.</li>
                    <li><strong>Same Wi‑Fi:</strong> Ensure your phone and PC are on the <strong>exact same Wi‑Fi network</strong> (not a guest network or different router).</li>
                    <li><strong>Check Vite output:</strong> In the terminal where you ran <code style={{ background: '#fed7aa', padding: '2px 6px', borderRadius: '4px' }}>npm run electron:dev</code>, look for "Network: http://..." — use that exact IP.</li>
                </ul>
            </div>

            <div className="card" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Note:</strong> Other devices must use the LAN URL shown above (with this PC’s IP). The dev server is configured to listen on all network interfaces.
            </div>
        </div>
    );
};

export default LANMultiComputerLogin;
