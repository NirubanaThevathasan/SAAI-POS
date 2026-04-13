import React, { createContext, useState, useContext, useEffect } from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';

const SettingsContext = createContext(null);

export const SettingsProvider = ({ children }) => {
    const [currency, setCurrency] = useState({ code: 'LKR', symbol: 'Rs.', exchange_rate: 1 });
    const [settings, setSettings] = useState({});

    // Load settings from DB
    const dbSettings = useLiveQuery(() => db.settings.toArray());
    const currencies = useLiveQuery(() => db.currencies.toArray());

    const updateSetting = async (key, value) => {
        await db.settings.put({ key, value });
    };

    const applyThemeColor = (color) => {
        document.documentElement.style.setProperty('--primary-color', color);
        // Add a slight transparency for hover if it's hex
        const hoverColor = color.length === 7 ? color + 'cc' : color;
        document.documentElement.style.setProperty('--primary-hover', hoverColor);
    };

    useEffect(() => {
        const initSettings = async () => {
            if (dbSettings) {
                const settingsObj = {};
                dbSettings.forEach(s => settingsObj[s.key] = s.value);
                setSettings(settingsObj);

                if (currencies) {
                    const hasLKR = currencies.find(c => c.code === 'LKR');
                    if (!hasLKR) {
                        await db.currencies.add({ code: 'LKR', name: 'Sri Lankan Rupee', symbol: 'Rs.', exchange_rate: 1 });
                    }

                    const activeCode = settingsObj.currency_code || 'LKR';
                    if (activeCode === 'INR' || !activeCode) {
                        await updateSetting('currency_code', 'LKR');
                        setCurrency({ code: 'LKR', symbol: 'Rs.', exchange_rate: 1 });
                    } else {
                        const activeCurr = currencies.find(c => c.code === activeCode);
                        if (activeCurr) setCurrency(activeCurr);
                    }
                }

                if (settingsObj.primary_color) {
                    applyThemeColor(settingsObj.primary_color);
                }
                if (settingsObj.theme) {
                    document.documentElement.setAttribute('data-theme', settingsObj.theme);
                }
            }
        };
        initSettings();
    }, [dbSettings, currencies]);

    const changeCurrency = async (code) => {
        await updateSetting('currency_code', code);
    };

    const changeThemeColor = async (color) => {
        applyThemeColor(color);
        await updateSetting('primary_color', color);
    };

    const toggleTheme = async () => {
        const current = settings.theme || 'light';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        await updateSetting('theme', next);
    };

    const formatPrice = (amount) => {
        const value = (amount || 0);
        return `${currency.symbol}${value.toFixed(2)}`;
    };

    return (
        <SettingsContext.Provider value={{
            settings,
            currency,
            changeCurrency,
            formatPrice,
            updateSetting,
            currencies: currencies || [],
            changeThemeColor,
            toggleTheme
        }}>
            {children}
        </SettingsContext.Provider>
    );
};

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (!context) {
        console.warn('useSettings must be used within SettingsProvider');
        // Return default values instead of null to prevent crashes
        return {
            settings: {},
            currency: { code: 'LKR', symbol: 'Rs.', exchange_rate: 1 },
            changeCurrency: async () => { },
            formatPrice: (amount) => `Rs.${(amount || 0).toFixed(2)}`,
            updateSetting: async () => { },
            currencies: [],
            changeThemeColor: async () => { },
            toggleTheme: async () => { }
        };
    }
    return context;
};
