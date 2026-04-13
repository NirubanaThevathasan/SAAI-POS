import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ProtectedRoute = ({ allowedRoles, permission }) => {
    const { user, hasPermission } = useAuth();

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    if (permission && !hasPermission(permission)) {
        return <div style={{
            padding: '50px',
            textAlign: 'center',
            color: '#e11d48',
            background: '#fff1f2',
            margin: '20px',
            borderRadius: '12px',
            border: '1px solid #fecdd3',
            fontWeight: '600'
        }}>
            <h2>Access Denied</h2>
            <p>You do not have the necessary permissions ({permission}) to view this page.</p>
        </div>;
    }

    // Fallback for legacy role checks if still used
    if (allowedRoles && !allowedRoles.includes(user.role) && user.role !== 'admin') {
        return <div style={{ padding: '20px', color: 'red' }}>Access Denied: Insufficient Role</div>;
    }

    return <Outlet />;
};

export default ProtectedRoute;
