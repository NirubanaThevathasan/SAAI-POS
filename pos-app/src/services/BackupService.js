import { db } from '../db';

/**
 * Exports all data from the IndexedDB database to a JSON object.
 * @returns {Promise<Object>} The backup object containing all table data.
 */
export const exportDatabase = async () => {
    const backup = {
        meta: {
            version: 1,
            timestamp: new Date().toISOString(),
            app: 'Enterprise POS',
        },
        data: {}
    };

    const tables = db.tables.map(table => table.name);

    for (const tableName of tables) {
        try {
            backup.data[tableName] = await db.table(tableName).toArray();
        } catch (error) {
            console.error(`Failed to export table ${tableName}:`, error);
        }
    }

    return backup;
};

/**
 * Imports data from a JSON backup object into the IndexedDB database.
 * @param {Object} backupData The backup data to import.
 * @returns {Promise<void>}
 */
export const importDatabase = async (backupData) => {
    if (!backupData || !backupData.data) {
        throw new Error('Invalid backup file: Missing data object.');
    }

    const tableNames = db.tables.map(t => t.name);

    // Use a transaction to ensure atomicity - either fully restores or rolls back (mostly)
    // Note: Dexie transactions for 'rw' lock the tables.
    await db.transaction('rw', tableNames, async () => {
        // 1. Clear all existing tables
        for (const tableName of tableNames) {
            await db.table(tableName).clear();
        }

        // 2. Populate with new data
        for (const tableName of Object.keys(backupData.data)) {
            // Only import tables that exist in our current schema to avoid errors
            if (tableNames.includes(tableName)) {
                const rows = backupData.data[tableName];
                if (Array.isArray(rows) && rows.length > 0) {
                    // We use bulkAdd for performance.
                    // If keys are auto-incremented and preserving IDs is important,
                    // bulkAdd honors the keys provided in the objects.
                    await db.table(tableName).bulkAdd(rows);
                }
            } else {
                console.warn(`Skipping unknown table in backup: ${tableName}`);
            }
        }
    });
};

/**
 * Triggers a file download for the backup data.
 */
export const downloadBackup = async () => {
    try {
        const data = await exportDatabase();
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const date = new Date().toISOString().split('T')[0];
        const time = new Date().toTimeString().split(' ')[0].replace(/:/g, '-');
        const filename = `pos-backup-${date}-${time}.json`;

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return true;
    } catch (error) {
        console.error('Export failed:', error);
        throw error;
    }
};
