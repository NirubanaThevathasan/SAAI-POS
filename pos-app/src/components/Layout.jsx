import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header'; // Ensure Header is used if it was there before
import Sidebar from './Sidebar';
import AIChatbot from './AIChatbot';

const Layout = () => {
    return (
        <div className="app-container">
            <Sidebar />
            <div className="main-content">
                <Header />
                <div className="page-content">
                    <Outlet />
                </div>
            </div>
            <AIChatbot />
        </div>
    );
};

export default Layout;
