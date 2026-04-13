// Preload script for Electron
// This file is loaded before the renderer process
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
    'electronAPI', {
    visionStart: (config) => ipcRenderer.invoke('vision:start', config),
    visionRunInference: () => ipcRenderer.invoke('vision:run-inference'),
    visionStop: () => ipcRenderer.invoke('vision:stop'),
    visionStatus: () => ipcRenderer.invoke('vision:status'),
    onVisionLog: (callback) => ipcRenderer.on('vision:log', callback),
    onVisionError: (callback) => ipcRenderer.on('vision:error', callback),
    onVisionStopped: (callback) => ipcRenderer.on('vision:stopped', callback)
}
);
