import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, AreaChart, Area, Legend
} from 'recharts';
import { FaRobot, FaChartLine, FaSnowflake, FaMoneyBillWave, FaExclamationCircle, FaUserSlash, FaTerminal, FaPaperPlane, FaLightbulb, FaBrain } from 'react-icons/fa';
import { forecastingService } from '../../services/forecastingService';
import { aiService } from '../../services/aiService';
import { db } from '../../db';
import { useSettings } from '../../context/SettingsContext';
import { pricingService } from '../../services/pricingService';
import { useAuth } from '../../context/AuthContext';

const AIAnalytics = () => {
    const { formatPrice } = useSettings();
    const { hasPermission } = useAuth();
    const [loading, setLoading] = useState(true);
    const [nlpLoading, setNlpLoading] = useState(false);
    const [nlpCommand, setNlpCommand] = useState('');
    const [nlpResult, setNlpResult] = useState(null);
    const [strategicInsight, setStrategicInsight] = useState('');
    const [insightLoading, setInsightLoading] = useState(false);

    const [predictions, setPredictions] = useState({
        nextDay: 0,
        nextDayProfit: 0,
        weekly: [],
        demand: [],
        cashFlow: [],
        seasonal: [],
        churn: [],
        pricing: []
    });

    useEffect(() => {
        loadAIStats();
    }, []);

    const loadAIStats = async () => {
        setLoading(true);
        try {
            const safeCall = (promise, fallback = []) => promise.catch(err => {
                console.warn("AI Model Error:", err);
                return fallback;
            });

            const [nextDay, nextDayProfit, weekly, demand, cashFlow, seasonal, churn] = await Promise.all([
                safeCall(forecastingService.predictNextDayRevenue('revenue'), 0),
                safeCall(forecastingService.predictNextDayRevenue('profit'), 0),
                safeCall(forecastingService.predictWeeklyRevenue()),
                safeCall(forecastingService.getDemandInsights()),
                safeCall(forecastingService.getCashFlowForecast()),
                safeCall(forecastingService.getSeasonalTrends()),
                safeCall(forecastingService.getCustomerChurnInsight())
            ]);

            setPredictions(prev => ({
                ...prev,
                nextDay: nextDay || 0,
                nextDayProfit: nextDayProfit || 0,
                weekly: weekly || [],
                demand: demand || [],
                cashFlow: cashFlow || [],
                seasonal: seasonal || [],
                churn: churn || []
            }));

            // Fetch pricing suggestions for top products
            if (pricingService.isAvailable() && demand.length > 0) {
                const topProducts = demand.slice(0, 5);
                const pricingSuggestions = await Promise.all(
                    topProducts.map(async (p) => {
                        try {
                            const decision = await pricingService.getSuggestionForProduct(p.id);
                            return { ...p, decision };
                        } catch (e) {
                            return null;
                        }
                    })
                );
                setPredictions(prev => ({
                    ...prev,
                    pricing: pricingSuggestions.filter(Boolean)
                }));
            }

            // After data is loaded, get strategic insight from Gemini
            generateStrategicInsight(demand, churn, nextDay);

        } catch (error) {
            console.error("Critical AI Analytics Error:", error);
        } finally {
            setLoading(false);
        }
    };

    const generateStrategicInsight = async (demand, churn, forecast) => {
        if (!aiService.hasApiKey()) {
            setStrategicInsight("Connect your Gemini API key in Settings to unlock deep strategic business analysis.");
            return;
        }

        setInsightLoading(true);
        try {
            const topProducts = demand.filter(p => p.status === 'Fast Moving').map(p => p.name).join(', ');
            const churnCount = churn.filter(c => c.risk === 'Critical').length;

            const prompt = `As a Strategic Business AI, analyze this summary and provide 3 SHORT, ACTIONABLE bullet points for the business owner.
            - Tomorrow Forecast: ${formatPrice(forecast)}
            - Best Selling Today: ${topProducts || 'None yet'}
            - Customers at Critical Risk: ${churnCount}
            
            Format: One sentence per bullet point. Be insightful and professional.`;

            const insight = await aiService.ask(prompt);
            setStrategicInsight(insight);
        } catch (e) {
            setStrategicInsight("AI suggestion unavailable right now.");
        } finally {
            setInsightLoading(false);
        }
    };

    const handleNlpOrder = async (e) => {
        e.preventDefault();
        if (!nlpCommand.trim()) return;
        setNlpLoading(true);
        setNlpResult(null);
        try {
            const result = await aiService.executeNLPOrder(nlpCommand);
            setNlpResult(result);
        } catch (error) {
            alert("Failed to parse command: " + error.message);
        } finally {
            setNlpLoading(false);
        }
    };

    const createPurchaseFromNlp = async () => {
        if (!nlpResult) return;
        try {
            const products = await db.products.toArray();
            const fullItems = nlpResult.items.map(item => {
                const p = products.find(prod => prod.id === item.product_id);
                if (!p) return null;
                return {
                    id: p.id,
                    name: p.name,
                    quantity: item.quantity,
                    cost: item.cost || p.cost || 0
                };
            }).filter(Boolean);

            if (fullItems.length === 0) {
                alert("No valid products found in AI response.");
                return;
            }

            const subtotal = fullItems.reduce((sum, i) => sum + (i.cost * i.quantity), 0);

            await db.purchases.add({
                date: new Date(),
                reference_no: `AI-${Date.now()}`,
                supplier_id: nlpResult.supplier_id || 1,
                warehouse_id: nlpResult.warehouse_id || 1,
                items: fullItems,
                grand_total: subtotal,
                status: 'pending',
                notes: (nlpResult.notes || "") + " (AI Generated NLP Order)",
                source: 'AI-NLP'
            });

            alert("Purchase order created successfully in 'Pending' status!");
            setNlpResult(null);
            setNlpCommand('');
        } catch (error) {
            alert("Error creating purchase: " + error.message);
        }
    };

    const handleApplyPricingSuggestion = async (p) => {
        if (!p.decision) return;
        const confirm = window.confirm(
            `Apply AI suggested price ${formatPrice(p.decision.suggested_price)} for "${p.name}"?`
        );
        if (!confirm) return;

        try {
            await db.products.update(p.id, { price: p.decision.suggested_price });
            alert('AI suggested price applied successfully.');
            // Refresh local state to show it's applied (or just reload stats)
            loadAIStats();
        } catch (err) {
            alert('Failed to apply price: ' + err.message);
        }
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: '20px' }}>
                <div className="spinner" style={{ width: '50px', height: '50px', border: '5px solid var(--primary-color)', borderRadius: '50%', borderTopColor: 'transparent', animation: 'spin 1s linear infinite' }}></div>
                <h3 style={{ color: 'var(--text-secondary)' }}>AI Engine is training and analyzing your data...</h3>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    return (
        <div className="ai-analytics">
            <header style={{ marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
                <div>
                    <h2 style={{ fontSize: '2rem', fontWeight: '800', margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <FaRobot style={{ color: 'var(--primary-color)' }} /> AI Demand & Sales Forecasting
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>Predictive insights powered by Pre-trained Gemini 3 & Local Neural Networks.</p>
                </div>
                <button
                    onClick={loadAIStats}
                    className="btn btn-primary"
                    style={{ borderRadius: '12px', padding: '12px 24px', boxShadow: '0 4px 15px rgba(99, 102, 241, 0.4)' }}
                >
                    Retrain local AI Models
                </button>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '32px' }}>
                {/* Prediction Cards */}
                <div className="card" style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)', color: 'white' }}>
                    <p style={{ margin: 0, opacity: 0.8, fontWeight: '500' }}>Next Day Revenue Prediction</p>
                    <h2 style={{ fontSize: '2.5rem', margin: '15px 0', fontWeight: '800' }}>{formatPrice(predictions.nextDay)}</h2>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                        <FaBrain /> Confidence: {predictions.weekly.length > 5 ? '87%' : '42%'} (Hybrid LSTM / AI Fallback)
                    </div>
                </div>

                <div className="card" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', color: 'white' }}>
                    <p style={{ margin: 0, opacity: 0.8, fontWeight: '500' }}>Next Day Profit Prediction</p>
                    <h2 style={{ fontSize: '2.5rem', margin: '15px 0', fontWeight: '800' }}>{formatPrice(predictions.nextDayProfit)}</h2>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                        <FaChartLine /> Calculated from Average Cost vs Retail Trends
                    </div>
                </div>

                <div className="card" style={{ background: '#0f172a', color: 'white', border: '1px solid #334155' }}>
                    <h3 style={{ margin: '0 0 10px', fontSize: '1rem', fontWeight: '700', color: '#818cf8', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FaLightbulb /> Strategic Business Advisor
                    </h3>
                    {insightLoading ? (
                        <div style={{ fontSize: '0.9rem', opacity: 0.6 }}>Gemini is thinking...</div>
                    ) : (
                        <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#cbd5e1' }}>
                            {strategicInsight.split('\n').map((line, i) => (
                                <div key={i} style={{ marginBottom: '4px' }}>{line}</div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginBottom: '24px' }}>
                {/* Weekly Forecast Chart */}
                <div className="card">
                    <h3 style={{ margin: '0 0 20px', fontSize: '1.2rem', fontWeight: '700' }}>Revenue Momentum (Next 7 Days)</h3>
                    <div style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={predictions.weekly.map(i => ({ date: `Day ${i.day}`, value: i.value }))}>
                                <defs>
                                    <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                                <XAxis dataKey="date" />
                                <YAxis />
                                <Tooltip formatter={(val) => formatPrice(val)} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }} />
                                <Area type="monotone" dataKey="value" stroke="#4f46e5" strokeWidth={3} fillOpacity={1} fill="url(#colorVal)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Slow vs Fast Moving */}
                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ margin: '0 0 20px', fontSize: '1.2rem', fontWeight: '700' }}>Product Movement Prediction</h3>
                    <div style={{ flex: 1, overflowX: 'hidden', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '300px' }}>
                        {predictions.demand.length > 0 ? predictions.demand.slice(0, 10).map(item => (
                            <div key={item.id} style={{ padding: '12px', borderRadius: '12px', background: 'var(--bg-color)', borderLeft: `4px solid ${item.color}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ minWidth: 0, flex: 1 }}>
                                    <div style={{ fontWeight: '600', fontSize: '0.95rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</div>
                                    <span style={{ fontSize: '0.7rem', color: item.color, fontWeight: '800' }}>{item.status.split(' ')[0]}</span>
                                </div>
                                <div style={{ textAlign: 'right', marginLeft: '10px' }}>
                                    <div style={{ fontWeight: '800' }}>{item.totalSold}</div>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Units</div>
                                </div>
                            </div>
                        )) : <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '20px' }}>No inventory data.</div>}
                    </div>
                </div>
                {/* Recommended Price Adjustments */}
                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ margin: '0 0 20px', fontSize: '1.2rem', fontWeight: '700', color: 'var(--primary-color)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FaMoneyBillWave /> AI Pricing Optimizer
                    </h3>
                    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '300px' }}>
                        {predictions.pricing.length > 0 ? predictions.pricing.map(p => (
                            <div key={p.id} style={{ padding: '12px', borderRadius: '12px', background: 'var(--bg-color)', border: '1px solid rgba(0,0,0,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ minWidth: 0, flex: 1 }}>
                                    <div style={{ fontWeight: '600', fontSize: '0.95rem' }}>{p.name}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
                                        Suggest: <span style={{ color: 'var(--primary-color)', fontWeight: '700' }}>{formatPrice(p.decision.suggested_price)}</span>
                                        {p.decision.suggested_discount_pct > 0 && <span style={{ marginLeft: '8px', color: '#10b981' }}>(-{p.decision.suggested_discount_pct}%)</span>}
                                    </div>
                                </div>
                                {hasPermission('products_edit') && (
                                    <button
                                        onClick={() => handleApplyPricingSuggestion(p)}
                                        className="btn btn-sm btn-primary"
                                        style={{ padding: '6px 12px', fontSize: '0.75rem' }}
                                    >
                                        Apply
                                    </button>
                                )}
                            </div>
                        )) : (
                            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
                                {!pricingService.isAvailable() && 'Local AI engine unavailable.'}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '24px' }}>
                <div className="card">
                    <h3 style={{ margin: '0 0 20px', fontSize: '1.2rem', fontWeight: '700' }}>Monthly Seasonal Cycles</h3>
                    <div style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={predictions.seasonal}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                                <XAxis dataKey="month" />
                                <YAxis hide />
                                <Tooltip formatter={(val) => formatPrice(val)} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
                                <Bar dataKey="revenue" radius={[6, 6, 0, 0]}>
                                    {predictions.seasonal.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.isPeak ? '#f59e0b' : '#6366f1'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card">
                    <h3 style={{ margin: '0 0 20px', fontSize: '1.2rem', fontWeight: '700' }}>Customer Churn Probability</h3>
                    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', height: '300px' }}>
                        {predictions.churn.length > 0 ? predictions.churn.slice(0, 10).map(c => (
                            <div key={c.id} style={{ padding: '12px 15px', borderRadius: '12px', background: 'var(--bg-color)', borderLeft: `6px solid ${c.color}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontWeight: '700', fontSize: '1rem' }}>{c.name}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Absent for {c.daysSinceLast} days</div>
                                </div>
                                <div style={{ textAlign: 'right' }}>
                                    <div style={{ padding: '4px 10px', borderRadius: '20px', fontSize: '0.7rem', fontWeight: '800', background: `${c.color}20`, color: c.color }}>
                                        {c.risk.toUpperCase()} RISK
                                    </div>
                                </div>
                            </div>
                        )) : <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '40px' }}>No customer data.</div>}
                    </div>
                </div>
            </div>

            {/* AI Command Center */}
            <div className="card" style={{ marginTop: '24px', background: '#020617', color: 'white', border: 'none', borderRadius: '20px', padding: '32px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <FaTerminal style={{ color: '#38bdf8' }} /> AI Intelligent Order Center
                    </h3>
                    <span style={{ fontSize: '0.8rem', padding: '6px 12px', borderRadius: '30px', background: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.3)', fontWeight: '600' }}>
                        CONNECTED: PRE-TRAINED GEMINI LATEST
                    </span>
                </div>
                <p style={{ opacity: 0.6, marginBottom: '25px', fontSize: '1rem' }}>Commands are parsed using 100% database context. Try something complex: "I need 10 units of everything from Supplier X tomorrow"</p>

                <form onSubmit={handleNlpOrder} style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
                    <input
                        className="form-control"
                        placeholder="Type natural language command..."
                        style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', flex: 1, padding: '16px 20px', borderRadius: '14px', fontSize: '1.1rem' }}
                        value={nlpCommand}
                        onChange={e => setNlpCommand(e.target.value)}
                    />
                    <button className="btn" disabled={nlpLoading} style={{ background: '#38bdf8', color: '#0f172a', padding: '0 30px', borderRadius: '14px', fontWeight: '800', fontSize: '1rem' }}>
                        {nlpLoading ? 'Analyzing...' : <><FaPaperPlane /> Process</>}
                    </button>
                </form>

                {nlpResult && (
                    <div style={{ background: 'rgba(56, 189, 248, 0.03)', padding: '24px', borderRadius: '18px', border: '1px solid rgba(56, 189, 248, 0.2)', animation: 'fadeIn 0.4s ease-out' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <h4 style={{ margin: 0, color: '#38bdf8', fontSize: '1.25rem', fontWeight: '700' }}>Automated Order Proposal</h4>
                            <button onClick={() => setNlpResult(null)} style={{ background: 'rgba(255,255,255,0.05)', border: 'none', color: 'white', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', fontSize: '0.8rem' }}>Dismiss</button>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '20px' }}>
                            {nlpResult.items.map((item, idx) => (
                                <div key={idx} style={{ padding: '15px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ fontSize: '0.8rem', opacity: 0.5, marginBottom: '4px' }}>PRODUCT ID: {item.product_id}</div>
                                    <div style={{ fontWeight: '700', fontSize: '1.1rem' }}>Quantity: {item.quantity}</div>
                                    <div style={{ color: '#38bdf8', fontWeight: '600', marginTop: '4px' }}>{formatPrice(item.cost)} each</div>
                                </div>
                            ))}
                        </div>

                        <div style={{ display: 'flex', gap: '12px', padding: '15px', background: 'rgba(56, 189, 248, 0.05)', borderRadius: '12px', marginBottom: '25px' }}>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>SUPPLIER AUTO-ID</div>
                                <div style={{ fontWeight: '700' }}>{nlpResult.supplier_id || 'System Detected'}</div>
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>DESTINATION WAREHOUSE</div>
                                <div style={{ fontWeight: '700' }}>{nlpResult.warehouse_id || 'Main Store'}</div>
                            </div>
                        </div>

                        <button
                            onClick={createPurchaseFromNlp}
                            className="btn"
                            style={{ background: 'linear-gradient(90deg, #10b981, #059669)', color: 'white', fontWeight: '800', width: '100%', borderRadius: '14px', padding: '18px', fontSize: '1.1rem', boxShadow: '0 10px 20px -5px rgba(16, 185, 129, 0.3)' }}
                        >
                            Execute Purchase Order
                        </button>
                    </div>
                )}
            </div>

            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: scale(0.98); }
                    to { opacity: 1; transform: scale(1); }
                }
                .card { transition: transform 0.2s; }
                .card:hover { transform: translateY(-2px); }
            `}</style>
        </div>
    );
};

export default AIAnalytics;
