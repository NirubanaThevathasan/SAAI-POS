import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '40px',
                    textAlign: 'center',
                    background: '#fff1f2',
                    borderRadius: '12px',
                    margin: '20px',
                    border: '2px solid #fecdd3'
                }}>
                    <h2 style={{ color: '#e11d48', marginBottom: '10px' }}>Something went wrong</h2>
                    <p style={{ color: '#991b1b', marginBottom: '20px' }}>
                        {this.state.error?.message || 'An unexpected error occurred'}
                    </p>
                    <details style={{ textAlign: 'left', marginBottom: '20px', background: '#fee2e2', padding: '15px', borderRadius: '8px' }}>
                        <summary style={{ cursor: 'pointer', fontWeight: '600', marginBottom: '10px' }}>Error Details</summary>
                        <pre style={{ fontSize: '0.85rem', overflow: 'auto' }}>
                            {this.state.error?.stack || this.state.error?.toString()}
                        </pre>
                    </details>
                    <button
                        className="btn btn-primary"
                        onClick={() => {
                            this.setState({ hasError: false, error: null });
                            window.location.reload();
                        }}
                    >
                        Reload Page
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;

