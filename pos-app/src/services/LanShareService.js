import { exportDatabase, importDatabase } from './BackupService';
import { db } from '../db';

function isElectron() {
    return typeof window !== 'undefined' && typeof require !== 'undefined';
}

function getIpcRenderer() {
    try {
        const { ipcRenderer } = require('electron');
        return ipcRenderer;
    } catch {
        return null;
    }
}

export async function publishMainPcSnapshot() {
    const ipc = getIpcRenderer();
    if (!ipc) {
        return { status: 'error', message: 'Not running in Electron main PC.' };
    }
    const backup = await exportDatabase();
    const json = JSON.stringify(backup);
    ipc.send('lan:backup:update', json);
    return { status: 'success', bytes: json.length };
}

export async function isLanShareEnabled() {
    try {
        const row = await db.settings.get('lan_share_enabled');
        return row?.value === true || row?.value === 'true';
    } catch {
        return false;
    }
}

export async function setLanShareEnabled(enabled) {
    await db.settings.put({ key: 'lan_share_enabled', value: !!enabled });
}

export function initLanSharePublisher() {
    if (!isElectron()) return;
    const ipc = getIpcRenderer();
    if (!ipc) return;

    // Publish immediately if enabled, then every 60 seconds
    const publishIfEnabled = async () => {
        const enabled = await isLanShareEnabled();
        if (!enabled) return;
        try {
            await publishMainPcSnapshot();
        } catch (e) {
            console.warn('LAN publish failed', e);
        }
    };

    publishIfEnabled();
    setInterval(publishIfEnabled, 60_000);
}

export async function syncFromMainPcBackup(backupUrl) {
    const res = await fetch(backupUrl, { cache: 'no-store' });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Failed to fetch backup (${res.status}): ${text}`);
    }
    const backup = await res.json();
    await importDatabase(backup);
}

