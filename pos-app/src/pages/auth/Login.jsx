import React, { useState } from 'react';
import { useAuth } from "../../context/AuthContext";
import { useNavigate } from 'react-router-dom';
import { FaLock, FaUser, FaExclamationCircle, FaShoppingCart, FaChartLine, FaBox, FaMoneyBillWave } from 'react-icons/fa';

const Login = () => {
    const [username, setUsername] = useState('admin');
    const [password, setPassword] = useState('123');
    const [error, setError] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsSubmitting(true);

        // Brief artificial delay for premium feel
        await new Promise(resolve => setTimeout(resolve, 800));

        const result = await login(username, password);
        if (result.success) {
            navigate('/');
        } else {
            setError(result.message);
            setIsSubmitting(false);
        }
    };

    return (
        <div style={{
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
            position: 'relative',
            overflow: 'hidden',
            fontFamily: "'Inter', sans-serif"
        }}>
            {/* Animated Background Visuals */}
            <div className="bg-visuals">
                <FloatingIcon icon={<FaShoppingCart />} top="15%" left="10%" color="var(--primary-color)" size="60px" delay="0s" duration="15s" />
                <FloatingIcon icon={<FaChartLine />} top="75%" left="15%" color="var(--success-color)" size="80px" delay="2s" duration="18s" />
                <FloatingIcon icon={<FaBox />} top="20%" right="15%" color="var(--warning-color)" size="50px" delay="1s" duration="20s" />
                <FloatingIcon icon={<FaMoneyBillWave />} top="80%" right="10%" color="var(--accent-color)" size="70px" delay="3s" duration="22s" />
                <FloatingIcon icon={<FaUser />} top="45%" left="5%" color="#f472b6" size="40px" delay="5s" duration="25s" />

                {/* Gradient Orbs */}
                <div style={{ position: 'absolute', width: '800px', height: '800px', background: 'radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%)', top: '-20%', right: '-20%', borderRadius: '50%' }}></div>
                <div style={{ position: 'absolute', width: '600px', height: '600px', background: 'radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%)', bottom: '-10%', left: '-10%', borderRadius: '50%' }}></div>
            </div>

            <div className="login-card" style={{
                width: '100%',
                maxWidth: '440px',
                background: 'rgba(255, 255, 255, 0.03)',
                backdropFilter: 'blur(24px)',
                WebkitBackdropFilter: 'blur(24px)',
                padding: '48px',
                borderRadius: '28px',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
                zIndex: 10
            }}>
                <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                    <div style={{
                        width: '72px',
                        height: '72px',
                        background: 'linear-gradient(135deg, var(--primary-hover) 0%, var(--primary-color) 100%)',
                        borderRadius: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 auto 24px',
                        boxShadow: '0 8px 20px rgba(0, 0, 0, 0.2)'
                    }}>
                        <FaLock style={{ fontSize: '28px', color: '#fff' }} />
                    </div>
                    <h1 style={{ color: '#fff', fontSize: '1.85rem', fontWeight: '800', marginBottom: '10px', letterSpacing: '-0.025em' }}>SAAI POS</h1>
                    <p style={{ color: 'rgba(255, 255, 255, 0.45)', fontSize: '0.95rem', fontWeight: '400' }}>Secure access to your business portal</p>
                </div>

                <form onSubmit={handleSubmit}>
                    {error && (
                        <div style={{
                            background: 'rgba(244, 63, 94, 0.12)',
                            border: '1px solid rgba(244, 63, 94, 0.25)',
                            color: '#fda4af',
                            padding: '14px 18px',
                            borderRadius: '14px',
                            fontSize: '0.875rem',
                            marginBottom: '28px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px',
                            animation: 'shake 0.4s cubic-bezier(.36,.07,.19,.97) both'
                        }}>
                            <FaExclamationCircle style={{ fontSize: '1.1rem' }} /> {error}
                        </div>
                    )}

                    <div style={{ marginBottom: '24px' }}>
                        <label style={{ display: 'block', color: 'rgba(255, 255, 255, 0.85)', fontSize: '0.875rem', fontWeight: '600', marginBottom: '10px' }}>Username</label>
                        <div style={{ position: 'relative' }}>
                            <FaUser style={{ position: 'absolute', left: '18px', top: '16px', color: 'rgba(255, 255, 255, 0.35)', fontSize: '1rem' }} />
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter your username"
                                required
                                style={{
                                    width: '100%',
                                    padding: '14px 18px 14px 50px',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '14px',
                                    color: '#fff',
                                    fontSize: '1rem',
                                    outline: 'none',
                                    transition: 'all 0.2s ease'
                                }}
                            />
                        </div>
                    </div>

                    <div style={{ marginBottom: '36px' }}>
                        <label style={{ display: 'block', color: 'rgba(255, 255, 255, 0.85)', fontSize: '0.875rem', fontWeight: '600', marginBottom: '10px' }}>Password</label>
                        <div style={{ position: 'relative' }}>
                            <FaLock style={{ position: 'absolute', left: '18px', top: '16px', color: 'rgba(255, 255, 255, 0.35)', fontSize: '1rem' }} />
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                required
                                style={{
                                    width: '100%',
                                    padding: '14px 18px 14px 50px',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '14px',
                                    color: '#fff',
                                    fontSize: '1rem',
                                    outline: 'none',
                                    transition: 'all 0.2s ease'
                                }}
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={isSubmitting}
                        style={{
                            width: '100%',
                            padding: '16px',
                            background: 'linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%)',
                            border: 'none',
                            borderRadius: '14px',
                            color: '#fff',
                            fontSize: '1rem',
                            fontWeight: '700',
                            cursor: isSubmitting ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '12px',
                            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        {isSubmitting ? (
                            <>
                                <div className="login-spinner"></div>
                                Authenticating...
                            </>
                        ) : 'Sign In'}
                    </button>

                    <div style={{ marginTop: '32px', textAlign: 'center' }}>
                    </div>
                </form>
            </div>

            <style>{`
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                @keyframes float {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50% { transform: translateY(-20px) rotate(5deg); }
                }
                @keyframes shake {
                    10%, 90% { transform: translate3d(-1px, 0, 0); }
                    20%, 80% { transform: translate3d(2px, 0, 0); }
                    30%, 50%, 70% { transform: translate3d(-4px, 0, 0); }
                    40%, 60% { transform: translate3d(4px, 0, 0); }
                }
                .login-spinner {
                    width: 20px;
                    height: 20px;
                    border: 3px solid rgba(255,255,255,0.3);
                    border-top-color: #fff;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }
                .login-card input:focus {
                    background: rgba(255, 255, 255, 0.08) !important;
                    border-color: var(--primary-color) !important;
                    box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.05) !important;
                }
                .login-card button:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.4);
                }
                .login-card button:active:not(:disabled) {
                    transform: translateY(0);
                }
            `}</style>
        </div>
    );
};

// Helper Component for Animated Icons
const FloatingIcon = ({ icon, top, left, right, color, size, delay, duration }) => (
    <div style={{
        position: 'absolute',
        top, left, right,
        color: color,
        fontSize: size,
        opacity: 0.15,
        animation: `float ${duration} ease-in-out infinite`,
        animationDelay: delay,
        zIndex: 1
    }}>
        {icon}
    </div>
);

export default Login;
