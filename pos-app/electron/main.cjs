const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const http = require('http');
const os = require('os');
const { spawn } = require('child_process');

let mainWindow;
let visionProcess = null;
let lanBackupServer = null;
let lanBackupJson = null; // stringified JSON
let lanBackupUpdatedAt = null; // ISO string

/**
 * Helper to run a Python script that reads JSON from stdin and prints JSON to stdout.
 * Used for AI-powered helpers (vision, pricing, etc.).
 * @param {string} pythonPath
 * @param {string} scriptPath
 * @param {object} payload
 * @param {string} cwd
 * @returns {Promise<any>}
 */
function runPythonJson(pythonPath, scriptPath, payload, cwd) {
    return new Promise((resolve, reject) => {
        const proc = spawn(pythonPath, [scriptPath], { cwd });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        proc.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        proc.on('error', (err) => {
            reject(err);
        });

        proc.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(stderr || `Python exited with code ${code}`));
            }
            try {
                const parsed = JSON.parse(stdout || '{}');
                if (parsed && parsed.error && !parsed.product_id) {
                    // Structured error coming from CLI
                    return reject(new Error(parsed.error));
                }
                resolve(parsed);
            } catch (e) {
                reject(new Error(`Failed to parse Python output: ${e.message || e}. Raw: ${stdout}`));
            }
        });

        try {
            proc.stdin.write(JSON.stringify(payload || {}));
            proc.stdin.end();
        } catch (e) {
            reject(e);
        }
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        title: "SAAI POS"
    });

    const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, '../dist/index.html')}`;
    mainWindow.loadURL(startUrl);

    mainWindow.on('closed', function () {
        // Kill vision process if running
        if (visionProcess) {
            visionProcess.kill();
            visionProcess = null;
        }
        mainWindow = null;
    });
}

// IPC Handlers for Python Vision System
ipcMain.handle('vision:start', async (event, config) => {
    try {
        const { videoSource, device, confThreshold } = config;
        const pythonPath = 'python';
        const scriptPath = path.join(__dirname, '../Smart Retail Monitor/src/inference.py');

        // Update config first
        const bridgePath = path.join(__dirname, '../Smart Retail Monitor/vision_bridge.py');
        if (!videoSource || typeof videoSource !== 'string') {
            return { status: 'error', message: 'Video file path is required. Webcam is not supported.' };
        }

        const configProcess = spawn(pythonPath, [
            bridgePath,
            'start',
            videoSource,
            'models/gpModel.pt',
            device || 'cpu',
            confThreshold || '0.25'
        ], {
            cwd: path.join(__dirname, '../Smart Retail Monitor')
        });

        return new Promise((resolve, reject) => {
            let output = '';

            configProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            configProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result);
                    } catch (e) {
                        resolve({ status: 'success', message: 'Configuration updated' });
                    }
                } else {
                    reject(new Error('Failed to configure vision system'));
                }
            });
        });
    } catch (error) {
        return { status: 'error', message: error.message };
    }
});

// IPC Handler for AI-powered Dynamic Pricing (Python)
ipcMain.handle('pricing:suggest', async (event, payload) => {
    try {
        const pythonPath = 'python';
        const scriptPath = path.join(__dirname, '../Smart Retail Monitor/pricing_cli.py');
        const cwd = path.join(__dirname, '../Smart Retail Monitor');

        const result = await runPythonJson(pythonPath, scriptPath, payload, cwd);
        return {
            status: 'success',
            decision: result
        };
    } catch (error) {
        console.error('pricing:suggest error', error);
        return {
            status: 'error',
            message: error.message || String(error)
        };
    }
});

ipcMain.handle('vision:run-inference', async (event) => {
    try {
        if (visionProcess) {
            return { status: 'error', message: 'Vision system is already running' };
        }

        const pythonPath = 'python';
        const scriptPath = path.join(__dirname, '../Smart Retail Monitor/src/inference.py');

        visionProcess = spawn(pythonPath, ['-m', 'src.inference'], {
            cwd: path.join(__dirname, '../Smart Retail Monitor')
        });

        visionProcess.stdout.on('data', (data) => {
            const str = data.toString();
            console.log(`Vision: ${str}`);
            // Parse progress line: [VISION_PROGRESS]percent|frame|total|eta_seconds|fps
            const progressMatch = str.match(/\[VISION_PROGRESS\]([\d.]+)\|(\d+)\|(\d+)\|(\d+)\|([\d.]+)/);
            if (progressMatch) {
                const [, percent, frame, total, etaSeconds, fps] = progressMatch;
                mainWindow.webContents.send('vision:progress', {
                    percent: parseFloat(percent),
                    frame: parseInt(frame, 10),
                    total: parseInt(total, 10),
                    etaSeconds: parseInt(etaSeconds, 10),
                    fps: parseFloat(fps)
                });
                return; // Don't add raw progress line to logs
            }
            mainWindow.webContents.send('vision:log', str);
        });

        visionProcess.stderr.on('data', (data) => {
            console.error(`Vision Error: ${data}`);
            mainWindow.webContents.send('vision:error', data.toString());
        });

        visionProcess.on('close', (code) => {
            console.log(`Vision process exited with code ${code}`);
            mainWindow.webContents.send('vision:stopped', code);
            mainWindow.webContents.send('vision:progress', null); // Clear progress
            visionProcess = null;
        });

        return { status: 'success', message: 'Vision system started' };
    } catch (error) {
        return { status: 'error', message: error.message };
    }
});

ipcMain.handle('vision:stop', async (event) => {
    if (visionProcess) {
        visionProcess.kill();
        visionProcess = null;
        return { status: 'success', message: 'Vision system stopped' };
    }
    return { status: 'error', message: 'Vision system is not running' };
});

ipcMain.handle('vision:status', async (event) => {
    return {
        status: 'success',
        running: visionProcess !== null
    };
});

// Receive exported DB snapshot from renderer (Electron main-PC)
ipcMain.on('lan:backup:update', (event, backupJsonString) => {
    if (typeof backupJsonString === 'string' && backupJsonString.length > 0) {
        lanBackupJson = backupJsonString;
        lanBackupUpdatedAt = new Date().toISOString();
    }
});

// LAN address for multi-computer login (prefer real adapters over virtual)
ipcMain.handle('lan:get-address', async () => {
    try {
        const interfaces = os.networkInterfaces();
        const virtualKeywords = ['virtualbox', 'vmware', 'hyper-v', 'vbox', 'vmnet', 'virtual', 'loopback'];
        const realIPs = [];
        const virtualIPs = [];
        
        for (const name of Object.keys(interfaces)) {
            const isVirtual = virtualKeywords.some(keyword => name.toLowerCase().includes(keyword));
            for (const iface of interfaces[name]) {
                if (iface.family === 'IPv4' && !iface.internal) {
                    if (isVirtual) {
                        virtualIPs.push({ ip: iface.address, name });
                    } else {
                        realIPs.push({ ip: iface.address, name });
                    }
                }
            }
        }
        
        // Return best IP (prefer real adapters) and all options
        const bestIp = realIPs.length > 0 ? realIPs[0].ip : (virtualIPs.length > 0 ? virtualIPs[0].ip : null);
        return { 
            lanIp: bestIp,
            allIPs: [...realIPs, ...virtualIPs].map(item => ({ ip: item.ip, name: item.name }))
        };
    } catch (e) {
        console.error('lan:get-address error', e);
    }
    return { lanIp: null, allIPs: [] };
});

function startLanBackupServer() {
    if (lanBackupServer) return;

    lanBackupServer = http.createServer((req, res) => {
        // Basic CORS for LAN clients (phone/browser)
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

        if (req.method === 'OPTIONS') {
            res.writeHead(204);
            res.end();
            return;
        }

        if (req.method !== 'GET') {
            res.writeHead(405, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'error', message: 'Method not allowed' }));
            return;
        }

        if (req.url === '/health') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status: 'ok',
                hasBackup: !!lanBackupJson,
                updatedAt: lanBackupUpdatedAt
            }));
            return;
        }

        if (req.url === '/backup.json') {
            if (!lanBackupJson) {
                res.writeHead(503, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    status: 'error',
                    message: 'No backup published yet. Open the POS on the main PC and click “Publish Main PC Data”.'
                }));
                return;
            }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(lanBackupJson);
            return;
        }

        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: 'Not found' }));
    });

    lanBackupServer.listen(4310, '0.0.0.0', () => {
        console.log('LAN backup server listening on http://0.0.0.0:4310');
    });
}

app.on('ready', () => {
    createWindow();
    startLanBackupServer();
});

app.on('window-all-closed', function () {
    if (visionProcess) {
        visionProcess.kill();
    }
    if (lanBackupServer) {
        try { lanBackupServer.close(); } catch (e) { /* ignore */ }
        lanBackupServer = null;
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', function () {
    if (mainWindow === null) {
        createWindow();
    }
});
