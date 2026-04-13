import { db } from '../db';
import { exportDatabase } from './BackupService';

const BACKUP_KEY = 'pos_auto_backup';
const BACKUP_TIMESTAMP_KEY = 'pos_backup_timestamp';
const BACKUP_INTERVAL = 5 * 60 * 1000; // 5 minutes in milliseconds

/**
 * Saves a backup to localStorage
 */
export const saveBackupToStorage = async () => {
    try {
        const backup = await exportDatabase();
        const backupStr = JSON.stringify(backup);
        
        // Check if localStorage has enough space (limit is usually 5-10MB)
        if (backupStr.length > 4 * 1024 * 1024) { // 4MB limit for safety
            console.warn('Backup too large for localStorage, skipping auto-backup');
            return false;
        }
        
        localStorage.setItem(BACKUP_KEY, backupStr);
        localStorage.setItem(BACKUP_TIMESTAMP_KEY, new Date().toISOString());
        console.log('Auto-backup saved to localStorage');
        return true;
    } catch (error) {
        console.error('Failed to save auto-backup:', error);
        return false;
    }
};

const RESTORE_FLAG_KEY = 'pos_restore_attempted';

/**
 * Restores data from localStorage backup if IndexedDB is empty
 */
export const restoreFromStorage = async () => {
    try {
        // Check if we've already attempted restore in this session
        if (sessionStorage.getItem(RESTORE_FLAG_KEY)) {
            console.log('Restore already attempted in this session');
            return false;
        }

        const backupStr = localStorage.getItem(BACKUP_KEY);
        if (!backupStr) {
            console.log('No backup found in localStorage');
            sessionStorage.setItem(RESTORE_FLAG_KEY, 'true');
            return false;
        }

        // Check if database has meaningful data
        const productCount = await db.products.count();
        const salesCount = await db.sales.count();
        const customerCount = await db.customers.count();
        
        // Only restore if database is empty or very minimal (just defaults)
        // Allow restore if there are less than 3 products and no sales/customers
        if (productCount > 2 || salesCount > 0 || customerCount > 0) {
            console.log('Database has data, skipping auto-restore');
            sessionStorage.setItem(RESTORE_FLAG_KEY, 'true');
            return false;
        }

        // Mark that we're attempting restore
        sessionStorage.setItem(RESTORE_FLAG_KEY, 'true');

        const backup = JSON.parse(backupStr);
        const backupService = await import('./BackupService');
        
        await backupService.importDatabase(backup);
        console.log('Data restored from localStorage backup');
        
        // Mark as restored - no notification, just restore silently
        sessionStorage.setItem('pos_restore_completed', 'true');
        
        return true;
    } catch (error) {
        console.error('Failed to restore from localStorage:', error);
        sessionStorage.setItem(RESTORE_FLAG_KEY, 'true');
        return false;
    }
};

/**
 * Gets the last backup timestamp
 */
export const getLastBackupTime = () => {
    const timestamp = localStorage.getItem(BACKUP_TIMESTAMP_KEY);
    return timestamp ? new Date(timestamp) : null;
};

/**
 * Initializes auto-backup system
 */
export const initAutoBackup = () => {
    // Save backup immediately on load
    saveBackupToStorage();

    // Set up periodic backups
    setInterval(() => {
        saveBackupToStorage();
    }, BACKUP_INTERVAL);

    // Save backup before page unload
    window.addEventListener('beforeunload', () => {
        // Use synchronous storage for beforeunload
        try {
            const backup = exportDatabase();
            backup.then(data => {
                const backupStr = JSON.stringify(data);
                if (backupStr.length < 4 * 1024 * 1024) {
                    localStorage.setItem(BACKUP_KEY, backupStr);
                    localStorage.setItem(BACKUP_TIMESTAMP_KEY, new Date().toISOString());
                }
            }).catch(err => console.error('Last backup failed:', err));
        } catch (error) {
            console.error('Failed to save final backup:', error);
        }
    });

    // Save backup on visibility change (when user switches tabs/apps)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            saveBackupToStorage();
        }
    });

    console.log('Auto-backup system initialized');
};

/**
 * Hook to auto-backup after database operations
 * Call this after any add/update/delete operations
 */
export const triggerAutoBackup = () => {
    // Debounce: only backup if last backup was more than 30 seconds ago
    const lastBackup = getLastBackupTime();
    const now = new Date();
    
    if (!lastBackup || (now - lastBackup) > 30000) { // 30 seconds
        saveBackupToStorage();
    }
};

