import React, { useEffect } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { SettingsProvider } from './context/SettingsContext';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';
import Login from './pages/auth/Login';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Products from './pages/Products';
import ProductRequests from './pages/ProductRequests';
import ShopNotes from './pages/ShopNotes';
import StockCounts from './pages/products/StockCounts';
import BarcodeLabels from './pages/products/BarcodeLabels';
import POS from './pages/POS';
import Sales from './pages/Sales';
import Deliveries from './pages/sales/Deliveries';
import Settings from './pages/Settings';
import People from './pages/People';
import Purchases from './pages/Purchases';
import Expenses from './pages/Expenses';
import Quotations from './pages/Quotations';
import Transfers from './pages/Transfers';
import Returns from './pages/Returns';
import Reports from './pages/Reports';
import AIAnalytics from './pages/reports/AIAnalytics';
import Calendar from './pages/Calendar';
import { initAutoBackup, restoreFromStorage } from './services/AutoBackupService';
import { initLanSharePublisher } from './services/LanShareService';
import './index.css';

function App() {
  useEffect(() => {
    // Initialize auto-backup system
    initAutoBackup();

    // If running on the main PC in Electron, optionally publish DB snapshots to LAN
    initLanSharePublisher();

    // DISABLED: Auto-restore to prevent continuous alerts
    // Users can manually restore from Settings → System Maintenance if needed
    // The auto-backup still works to save data, but restore is manual only
  }, []);

  return (
    <AuthProvider>
      <SettingsProvider>
        <HashRouter>
          <Routes>
            <Route path="/login" element={<Login />} />

            <Route path="/" element={<ProtectedRoute />} >
              <Route element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route element={<ProtectedRoute permission="pos_access" />}>
                  <Route path="pos" element={
                    <ErrorBoundary>
                      <POS />
                    </ErrorBoundary>
                  } />
                </Route>

                <Route element={<ProtectedRoute permission="products_view" />}>
                  <Route path="products" element={<Products />} />
                  <Route path="requests" element={<ProductRequests />} />
                  <Route path="notes" element={<ShopNotes />} />
                  <Route path="/products/stock-counts" element={<StockCounts />} />
                  <Route path="/products/barcodes" element={<BarcodeLabels />} />
                </Route>

                <Route element={<ProtectedRoute permission="sales_view" />}>
                  <Route path="sales" element={<Sales />} />
                  <Route path="/sales/deliveries" element={<Deliveries />} />
                </Route>

                <Route element={<ProtectedRoute permission="purchases_view" />}>
                  <Route path="purchases" element={<Purchases />} />
                </Route>

                <Route element={<ProtectedRoute permission="expenses_view" />}>
                  <Route path="expenses" element={<Expenses />} />
                </Route>

                <Route element={<ProtectedRoute permission="quotations_view" />}>
                  <Route path="quotations" element={<Quotations />} />
                </Route>

                <Route element={<ProtectedRoute permission="transfers_view" />}>
                  <Route path="transfers" element={<Transfers />} />
                </Route>

                <Route element={<ProtectedRoute permission="returns_view" />}>
                  <Route path="returns" element={<Returns />} />
                </Route>

                <Route element={<ProtectedRoute permission="customers_view" />}>
                  <Route path="customers" element={<People />} />
                </Route>

                <Route element={<ProtectedRoute permission="view_reports" />}>
                  <Route path="reports" element={<Reports />} />
                  <Route path="ai-analytics" element={<AIAnalytics />} />
                </Route>

                <Route element={<ProtectedRoute permission="manage_settings" />}>
                  <Route path="settings" element={<Settings />} />
                </Route>

                <Route path="calendar" element={<Calendar />} />

              </Route>
            </Route>
          </Routes>
        </HashRouter>
      </SettingsProvider>
    </AuthProvider>
  );
}

export default App;
