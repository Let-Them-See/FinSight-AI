'use client'
import './globals.css'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const NAV = [
    { href: '/', icon: '🏠', label: 'Overview' },
    { href: '/credit-risk', icon: '📊', label: 'Module 1 — Credit Risk' },
    { href: '/segmentation', icon: '👥', label: 'Module 2 — Segmentation' },
    { href: '/sentiment', icon: '📰', label: 'Module 3 — Sentiment' },
    { href: '/forecasting', icon: '📈', label: 'Module 4 — Forecasting' },
]

function Sidebar() {
    const path = usePathname()
    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <span className="brand-icon">₹</span>
                <div className="brand-name">FinSight AI</div>
                <div className="brand-bar" />
                <div className="brand-sub">Financial Intelligence Platform</div>
            </div>
            <nav className="sidebar-nav">
                <div className="nav-label">Select Module</div>
                {NAV.map(n => (
                    <Link
                        key={n.href}
                        href={n.href}
                        className={`nav-link${path === n.href ? ' active' : ''}`}
                    >
                        <span>{n.icon}</span>
                        <span>{n.label}</span>
                    </Link>
                ))}
            </nav>
            <div className="sidebar-footer">
                <strong style={{ color: 'var(--accent)' }}>FinSight AI v1.1</strong><br />
                Enterprise Analytics Suite<br />
                Indian BFSI Context<br /><br />
                <em>Powered by scikit-learn,<br />XGBoost, Prophet &amp; VADER</em>
            </div>
        </aside>
    )
}

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <head>
                <meta charSet="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>FinSight AI | Financial Intelligence Platform</title>
                <meta name="description" content="Multi-Dimensional Financial Intelligence Platform — Credit Risk, Customer Segmentation, News Sentiment, Forecasting" />
                <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>₹</text></svg>" />
            </head>
            <body>
                <div className="layout">
                    <Sidebar />
                    <main className="main">{children}</main>
                </div>
            </body>
        </html>
    )
}
