// Vision API Bridge for Renderer Process
// This file should be imported in your React app to access Electron IPC

const { ipcRenderer } = require('electron');

if (typeof window !== 'undefined') {
    window.electronAPI = {
        visionStart: (config) => ipcRenderer.invoke('vision:start', config),
        visionRunInference: () => ipcRenderer.invoke('vision:run-inference'),
        visionStop: () => ipcRenderer.invoke('vision:stop'),
        visionStatus: () => ipcRenderer.invoke('vision:status'),
        getLanAddress: () => ipcRenderer.invoke('lan:get-address'),
        onVisionLog: (callback) => ipcRenderer.on('vision:log', callback),
        onVisionError: (callback) => ipcRenderer.on('vision:error', callback),
        onVisionStopped: (callback) => ipcRenderer.on('vision:stopped', callback),
        onVisionProgress: (callback) => ipcRenderer.on('vision:progress', callback),
        // AI-powered pricing bridge
        pricingSuggest: (payload) => ipcRenderer.invoke('pricing:suggest', payload),
    };
}

module.exports = window.electronAPI;
