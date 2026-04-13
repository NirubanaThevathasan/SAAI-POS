import React, { createContext, useState, useContext, useEffect } from 'react';
import { db } from '../db';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const initAuth = async () => {
            // Force update groups to ensure permissions are always up to date
            // This fixes the 'White Screen' issue if permissions were missing or old
            await db.user_groups.put({
                id: 1, name: 'Admin', permissions: {
                    products_view: true, products_add: true, products_edit: true, products_delete: true,
                    sales_view: true, sales_add: true, sales_edit: true, sales_delete: true,
                    purchases_view: true, purchases_add: true, purchases_edit: true, purchases_delete: true,
                    expenses_view: true, expenses_add: true, expenses_edit: true, expenses_delete: true,
                    customers_view: true, customers_add: true, customers_edit: true, customers_delete: true,
                    suppliers_view: true, suppliers_add: true, suppliers_edit: true, suppliers_delete: true,
                    transfers_view: true, transfers_add: true, transfers_edit: true, transfers_delete: true,
                    quotations_view: true, quotations_add: true, quotations_edit: true, quotations_delete: true,
                    returns_view: true, returns_add: true, returns_edit: true, returns_delete: true,
                    users_view: true, users_add: true, users_edit: true, users_delete: true,
                    pos_access: true, pos_discount: true, pos_price_edit: true, pos_hold_sale: true,
                    pos_returns: true, pos_manual_item: true, pos_clear_cart: true,
                    view_reports: true, manage_settings: true
                }
            });

            await db.user_groups.put({
                id: 2, name: 'Staff', permissions: {
                    products_view: true, products_add: true, products_edit: true,
                    sales_view: true, sales_add: true,
                    customers_view: true, customers_add: true,
                    quotations_view: true, quotations_add: true,
                    suppliers_view: true, suppliers_add: false,
                    pos_access: true, pos_hold_sale: true, pos_manual_item: true
                }
            });

            await db.user_groups.put({
                id: 3, name: 'Cashier', permissions: {
                    products_view: true,
                    sales_view: true,
                    pos_access: true,
                    pos_hold_sale: true,
                    customers_view: true
                }
            });

            const userCount = await db.users.count();
            if (userCount === 0) {
                await db.users.bulkAdd([
                    {
                        username: 'admin',
                        password: '123',
                        email: 'admin@pos.com',
                        group_id: 1
                    },
                    {
                        username: 'admin_main',
                        password: 'admin123',
                        email: 'main_admin@pos.com',
                        group_id: 1
                    }
                ]);
                console.log('Seeded default admin users and groups');
            }

            const adminMain = await db.users.where('username').equals('admin_main').first();
            if (!adminMain) {
                await db.users.add({
                    username: 'admin_main',
                    password: 'admin123',
                    email: 'main_admin@pos.com',
                    group_id: 1
                });
            }

            // Check for persisted session
            const storedUser = localStorage.getItem('pos_user');
            if (storedUser) {
                setUser(JSON.parse(storedUser));
            }
            setLoading(false);
        };
        initAuth();
    }, []);

    const login = async (username, password) => {
        const foundUser = await db.users.where('username').equals(username).first();

        if (foundUser && foundUser.password === password) {
            // Fetch group permissions
            const group = await db.user_groups.get(parseInt(foundUser.group_id));
            const roleName = group ? group.name.toLowerCase() : (['admin', 'admin_main'].includes(foundUser.username) ? 'admin' : 'staff');
            const userData = {
                ...foundUser,
                role: roleName,
                permissions: group ? group.permissions : (roleName === 'admin' ? { all: true } : {})
            };
            setUser(userData);
            localStorage.setItem('pos_user', JSON.stringify(userData));
            return { success: true };
        }
        return { success: false, message: 'Invalid credentials' };
    };

    const logout = () => {
        setUser(null);
        localStorage.removeItem('pos_user');
    };

    const refreshPermissions = async () => {
        if (!user) return;
        const group = await db.user_groups.get(parseInt(user.group_id));
        if (group) {
            const updatedUser = { ...user, permissions: group.permissions };
            setUser(updatedUser);
            localStorage.setItem('pos_user', JSON.stringify(updatedUser));
        }
    };

    const hasPermission = (permission) => {
        if (!user) return false;
        if (user.role === 'admin') return true; // Special case for admin override if needed, though permission object should be enough
        return !!user.permissions?.[permission];
    };

    return (
        <AuthContext.Provider value={{ user, login, logout, loading, hasPermission, refreshPermissions }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        console.warn('useAuth must be used within AuthProvider');
        // Return default values instead of null to prevent crashes
        return {
            user: null,
            login: async () => ({ success: false, message: 'Not initialized' }),
            logout: () => { },
            loading: false,
            hasPermission: () => true, // Allow access by default if context not available
            refreshPermissions: async () => { }
        };
    }
    return context;
};
