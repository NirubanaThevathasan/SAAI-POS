import React, { useState, useRef, useEffect } from 'react';
import { IoSparkles, IoSend, IoClose, IoChatbubblesOutline } from 'react-icons/io5';
import { aiService } from '../services/aiService';
import './AIChatbot.css';

const AIChatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'ai', text: 'Hello! I am your SAAI POS Assistant. Ask me anything about your business stats, products, or customers!' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [apiKey, setApiKey] = useState('');
    const [needsApiKey, setNeedsApiKey] = useState(!aiService.hasApiKey());

    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const [showSuggestions, setShowSuggestions] = useState(true);
    const suggestions = [
        "What are my total sales today?",
        "Which items are low in stock?",
        "List my top customers.",
        "What is my current net profit?",
        "Summarize last 5 transactions."
    ];

    const handleSend = async (overrideInput = null) => {
        const userMsg = (overrideInput || input).trim();
        if (!userMsg || isLoading) return;

        setInput('');
        setShowSuggestions(false); // Hide suggestions once chatting starts
        setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
        setIsLoading(true);

        try {
            const aiResponse = await aiService.ask(userMsg, messages.slice(1));
            setMessages(prev => [...prev, { role: 'ai', text: aiResponse }]);

            // If user said "hi", show suggestions again
            if (userMsg.toLowerCase() === 'hi' || userMsg.toLowerCase() === 'hello') {
                setShowSuggestions(true);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', text: `Error: ${error.message}. Please check your API key.` }]);
        } finally {
            setIsLoading(false);
        }
    };

    const saveApiKey = () => {
        if (apiKey.trim()) {
            aiService.setApiKey(apiKey.trim());
            setNeedsApiKey(false);
        }
    };

    return (
        <div className="ai-chatbot-container">
            {!isOpen && (
                <button className="ai-toggle-button" onClick={() => setIsOpen(true)}>
                    <IoSparkles />
                </button>
            )}

            {isOpen && (
                <div className="ai-chat-window">
                    <div className="ai-chat-header">
                        <h3><IoSparkles /> Business AI</h3>
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button
                                className="ai-reset-btn"
                                title="Reset API Key"
                                onClick={() => {
                                    aiService.clearApiKey();
                                    setNeedsApiKey(true);
                                }}
                                style={{ background: 'none', border: 'none', color: '#fff', opacity: 0.8, cursor: 'pointer' }}
                            >
                                <IoChatbubblesOutline />
                            </button>
                            <button className="ai-close-btn" onClick={() => setIsOpen(false)}>
                                <IoClose />
                            </button>
                        </div>
                    </div>

                    <div className="ai-chat-messages">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`ai-message ${msg.role}`}>
                                {msg.text}
                            </div>
                        ))}
                        {isLoading && (
                            <div className="ai-message ai">
                                <div className="ai-loading">
                                    <div className="ai-dot"></div>
                                    <div className="ai-dot"></div>
                                    <div className="ai-dot"></div>
                                </div>
                            </div>
                        )}
                        {showSuggestions && !isLoading && !needsApiKey && (
                            <div className="ai-suggestions-container">
                                {suggestions.map((s, i) => (
                                    <button 
                                        key={i} 
                                        className="ai-suggestion-chip"
                                        onClick={() => handleSend(s)}
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {needsApiKey && (
                        <div className="ai-api-key-overlay">
                            <IoChatbubblesOutline size={40} color="#6366f1" />
                            <h4>Set Gemini API Key</h4>
                            <p>To use AI features, please provide a Gemini API key. It's stored locally on your device.</p>
                            <input
                                type="password"
                                className="ai-chat-input"
                                placeholder="Enter API Key..."
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                            />
                            <button
                                className="ai-send-btn"
                                style={{ width: '100%', marginTop: '10px' }}
                                onClick={saveApiKey}
                            >
                                Save Key
                            </button>
                            <a
                                href="https://aistudio.google.com/app/apikey"
                                target="_blank"
                                rel="noreferrer"
                                style={{ fontSize: '12px', marginTop: '10px', color: '#6366f1' }}
                            >
                                Get a free key here
                            </a>
                        </div>
                    )}

                    <div className="ai-chat-input-area">
                        <input
                            type="text"
                            className="ai-chat-input"
                            placeholder="Ask about sales, products..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            disabled={needsApiKey}
                        />
                        <button
                            className="ai-send-btn"
                            onClick={() => handleSend()}
                            disabled={needsApiKey || !input.trim()}
                        >
                            <IoSend />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIChatbot;
