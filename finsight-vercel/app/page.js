'use client'

const MODULES = [
    {
        icon: '📊', title: 'Module 1: Credit Risk Scoring',
        desc: 'Predict loan default probability using XGBoost · Random Forest · Logistic Regression. Calibrated to HDFC Bank / Bajaj Finserv NBFC norms with CIBIL score integration.',
        meta: 'Metrics: ROC-AUC · F1-Score · Confusion Matrix',
    },
    {
        icon: '👥', title: 'Module 2: Customer Segmentation',
        desc: 'K-Means clustering to identify 4 distinct customer archetypes: Mass Market, Urban Aspirant, Affluent Investor, and Senior Wealth Preserver.',
        meta: 'Metrics: Silhouette Score · Inertia · t-SNE Visualisation',
    },
    {
        icon: '📰', title: 'Module 3: Sentiment Analysis',
        desc: 'Score Indian financial news headlines (Economic Times, Mint, Business Standard) using VADER NLP — classify as Bullish / Neutral / Bearish.',
        meta: 'Output: Compound Score · Sentiment Label · Market Signal',
    },
    {
        icon: '📈', title: 'Module 4: Financial Forecasting',
        desc: 'Prophet-based time series forecasting of monthly loan disbursements (₹ Crores) with Indian fiscal year seasonality and 12-month projections.',
        meta: 'Metrics: MAPE · RMSE · R² · Forecast Horizon',
    },
]

export default function OverviewPage() {
    return (
        <>
            {/* Hero */}
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.6rem' }}>
                    <span style={{ fontSize: '3.2rem', filter: 'drop-shadow(0 0 12px rgba(255,177,98,0.6))', color: 'var(--accent)' }}>₹</span>
                    <div>
                        <h1>FinSight AI</h1>
                        <p>Multi-Dimensional Financial Intelligence Platform</p>
                    </div>
                </div>
                <div className="brand-bar" style={{ width: 80, marginLeft: 0 }} />
                <p className="sub">Enterprise Analytics Suite &nbsp;|&nbsp; Indian BFSI Context &nbsp;|&nbsp; TCS / HDFC / Infosys Grade</p>
            </div>

            {/* KPI cards */}
            <div className="metric-grid">
                {[
                    { label: '📊 ML Modules', value: '4', delta: 'Credit Risk · Segmentation · NLP · Forecasting' },
                    { label: '🗂️ Training Data', value: '35,000+', delta: 'Rows across all modules' },
                    { label: '🤖 Models Built', value: '6+', delta: 'Classifiers · Clusters · Prophet' },
                    { label: '🎯 Top Model AUC', value: '~0.89', delta: 'XGBoost — Credit Risk' },
                ].map(m => (
                    <div className="metric-card" key={m.label}>
                        <div className="label">{m.label}</div>
                        <div className="value">{m.value}</div>
                        <div className="delta">{m.delta}</div>
                    </div>
                ))}
            </div>

            <hr />
            <h2 style={{ color: 'var(--red)', marginBottom: '1rem', fontSize: '1.1rem', fontWeight: 700 }}>🗂️ Platform Modules</h2>

            <div className="col-2">
                {MODULES.map(m => (
                    <div className="card" key={m.title}>
                        <h3>{m.icon} {m.title}</h3>
                        <p>{m.desc}</p>
                        <p className="card-meta"><b>Metrics:</b> {m.meta.replace('Metrics: ', '').replace('Output: ', '')}</p>
                    </div>
                ))}
            </div>

            <hr />
            <div className="info-box">
                <b style={{ color: 'var(--accent)' }}>📌 How to Use This Dashboard</b><br /><br />
                1. Use the <b>left sidebar</b> to navigate between modules.<br />
                2. Each module has an <b>interactive prediction panel</b> — enter inputs and click to score.<br />
                3. Charts update in real-time based on your inputs.<br />
                4. Data is from the <b>synthetic Indian BFSI dataset</b> (15,000 rows per module).<br />
                5. All ML scoring runs <b>directly in your browser</b> — no server needed.
            </div>
        </>
    )
}
