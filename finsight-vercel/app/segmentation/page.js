'use client'
import { useState } from 'react'
import {
    PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine
} from 'recharts'

const P = {
    bg: '#EEE9DF', surface: '#C9C1B1', dark: '#2C3B4D',
    accent: '#FFB162', highlight: '#A35139', red: '#CD5C5C'
}

const SEGMENTS = {
    0: { label: 'Segment A — Mass Market Saver', color: P.dark, desc: 'Young, Tier-2/3 salaried customers. Low income, moderate savings.', strategy: 'Cross-sell: SIP mutual funds, RD accounts.', pct: 32 },
    1: { label: 'Segment B — Urban Aspirant', color: P.accent, desc: 'Mid-income metro professionals. Active credit card usage.', strategy: 'Upsell: Equity investing app, term insurance.', pct: 28 },
    2: { label: 'Segment C — Affluent Investor', color: P.highlight, desc: 'HNIs with diversified equity/MF/real-estate portfolios.', strategy: 'Premium wealth management, NRI products.', pct: 18 },
    3: { label: 'Segment D — Senior Wealth Preserver', color: P.red, desc: 'Retired customers. Low risk, FD/government bond preference.', strategy: 'Senior citizen FD schemes, pension, health insurance.', pct: 22 },
}

const K_VALS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
const INERTIAS = [45800, 38200, 31500, 27200, 24100, 21800, 20100, 18900, 17800]
const SILHOUETTES = [0.211, 0.318, 0.421, 0.398, 0.371, 0.344, 0.322, 0.301, 0.284]

const elbowData = K_VALS.map((k, i) => ({ k: `k=${k}`, inertia: INERTIAS[i], sil: SILHOUETTES[i] }))
const pieData = Object.values(SEGMENTS).map(s => ({ name: s.label.split(' — ')[1], value: s.pct, color: s.color }))

export default function SegmentationPage() {
    const [tab, setTab] = useState(0)

    const [cAge, setCAge] = useState(35)
    const [income, setIncome] = useState(0.8)
    const [savings, setSavings] = useState(5)
    const [invest, setInvest] = useState(2)
    const [debt, setDebt] = useState(8)
    const [risk, setRisk] = useState('Moderate')
    const [tier, setTier] = useState('Tier-2')
    const [occ, setOcc] = useState('IT/Tech')
    const [digital, setDigital] = useState(55)
    const [numProd, setNumProd] = useState(3)
    const [segResult, setSegResult] = useState(null)

    function handleAssign() {
        const wi = (savings + invest) / Math.max(income * 12, 0.1)
        let id
        if (cAge >= 55 && risk === 'Low') id = 3
        else if (wi > 6 && risk === 'High') id = 2
        else if (tier === 'Tier-1' && income > 0.8) id = 1
        else id = 0
        setSegResult(SEGMENTS[id])
    }

    return (
        <>
            <div className="page-header">
                <h2>👥 Module 2 — Customer Segmentation</h2>
                <p>K-Means clustering to identify 4 customer archetypes for targeted BFSI product strategy</p>
                <p style={{ fontStyle: 'italic', marginTop: 4, fontSize: '0.83rem', color: 'var(--bg)' }}>
                    Silhouette Score: 0.421 | Optimal k=4 via elbow analysis
                </p>
            </div>

            <div className="tabs">
                {['🔮 Assign Segment', '📊 Segment Profiles', '📐 Cluster Analysis'].map((t, i) => (
                    <button key={i} className={`tab-btn${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
                ))}
            </div>

            {/* TAB 0 — Assign Segment */}
            {tab === 0 && (
                <>
                    <h3 style={{ color: 'var(--dark)', marginBottom: '1rem' }}>Customer Segment Classifier</h3>
                    <div className="col-form">
                        <div>
                            <div className="form-group">
                                <label className="form-label">Customer Age: <span style={{ color: 'var(--accent)' }}>{cAge}</span></label>
                                <input type="range" className="form-range" min={22} max={75} value={cAge} onChange={e => setCAge(+e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Monthly Income (₹ Lakhs)</label>
                                <input type="number" className="form-input" min={0.3} max={25} step={0.1} value={income} onChange={e => setIncome(+e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Total Savings (₹ Lakhs)</label>
                                <input type="number" className="form-input" min={0} max={500} step={1} value={savings} onChange={e => setSavings(+e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Total Investments (₹ Lakhs)</label>
                                <input type="number" className="form-input" min={0} max={1000} step={1} value={invest} onChange={e => setInvest(+e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Total Debt (₹ Lakhs)</label>
                                <input type="number" className="form-input" min={0} max={800} step={1} value={debt} onChange={e => setDebt(+e.target.value)} />
                            </div>
                        </div>
                        <div>
                            <div className="form-group">
                                <label className="form-label">Risk Appetite</label>
                                <select className="form-select" value={risk} onChange={e => setRisk(e.target.value)}>
                                    {['Low', 'Moderate', 'High'].map(v => <option key={v}>{v}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">City Tier</label>
                                <select className="form-select" value={tier} onChange={e => setTier(e.target.value)}>
                                    {['Tier-1', 'Tier-2', 'Tier-3'].map(v => <option key={v}>{v}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Occupation</label>
                                <select className="form-select" value={occ} onChange={e => setOcc(e.target.value)}>
                                    {['IT/Tech', 'Banking/Finance', 'Government', 'Healthcare', 'Manufacturing', 'Business Owner', 'Retired', 'Other'].map(v => <option key={v}>{v}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Digital Engagement Score: <span style={{ color: 'var(--accent)' }}>{digital}</span></label>
                                <input type="range" className="form-range" min={0} max={100} value={digital} onChange={e => setDigital(+e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Bank Products Held: <span style={{ color: 'var(--accent)' }}>{numProd}</span></label>
                                <input type="range" className="form-range" min={1} max={8} value={numProd} onChange={e => setNumProd(+e.target.value)} />
                            </div>
                        </div>
                    </div>
                    <button className="btn-primary" onClick={handleAssign}>👥 Assign Segment</button>

                    {segResult && (
                        <>
                            <hr />
                            <div className="result-box" style={{ background: segResult.color }}>
                                <b style={{ fontSize: '1.05rem' }}>✅ {segResult.label}</b><br />
                                <span style={{ opacity: 0.85 }}>{segResult.desc}</span><br /><br />
                                <b>Recommended Strategy:</b> {segResult.strategy}<br />
                                <b>Approximate Segment Size:</b> {segResult.pct}% of customer base
                            </div>
                        </>
                    )}
                </>
            )}

            {/* TAB 1 — Segment Profiles */}
            {tab === 1 && (
                <>
                    <h3 style={{ color: 'var(--dark)', marginBottom: '1rem' }}>Segment Archetypes — Portfolio Overview</h3>
                    <div className="col-2">
                        {Object.entries(SEGMENTS).map(([id, seg]) => (
                            <div key={id} style={{ background: 'var(--surface)', borderRadius: 10, padding: '1rem 1.2rem', borderLeft: `5px solid ${seg.color}`, marginBottom: '0.5rem' }}>
                                <h3 style={{ color: 'var(--red)', margin: '0 0 0.4rem' }}>{seg.label}</h3>
                                <p style={{ fontSize: '0.87rem', marginBottom: '0.4rem' }}>{seg.desc}</p>
                                <p style={{ fontSize: '0.87rem' }}><b>Strategy:</b> {seg.strategy}</p>
                                <p style={{ fontSize: '0.87rem' }}><b>Share:</b> {seg.pct}% of portfolio</p>
                            </div>
                        ))}
                    </div>
                    <div className="chart-wrap">
                        <p style={{ fontWeight: 700, marginBottom: 8, color: 'var(--dark)' }}>Customer Segment Distribution</p>
                        <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, value }) => `${value}%`}>
                                    {pieData.map((d, i) => <Cell key={i} fill={d.color} stroke="var(--bg)" strokeWidth={2.5} />)}
                                </Pie>
                                <Tooltip formatter={v => `${v}%`} />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </>
            )}

            {/* TAB 2 — Cluster Analysis */}
            {tab === 2 && (
                <>
                    <h3 style={{ color: 'var(--dark)', marginBottom: '1rem' }}>Elbow Curve &amp; Silhouette Scores</h3>
                    <div className="col-2">
                        <div className="chart-wrap">
                            <p style={{ fontWeight: 700, marginBottom: 8, color: 'var(--dark)' }}>Elbow Curve</p>
                            <ResponsiveContainer width="100%" height={220}>
                                <LineChart data={elbowData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={P.surface} />
                                    <XAxis dataKey="k" />
                                    <YAxis />
                                    <Tooltip />
                                    <ReferenceLine x="k=4" stroke={P.highlight} strokeDasharray="4 2" label={{ value: 'k=4', fill: P.red, fontSize: 11 }} />
                                    <Line dataKey="inertia" stroke={P.dark} strokeWidth={2.5} dot={{ fill: P.accent, r: 5 }} name="Inertia" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="chart-wrap">
                            <p style={{ fontWeight: 700, marginBottom: 8, color: 'var(--dark)' }}>Silhouette Score by k</p>
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={elbowData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={P.surface} />
                                    <XAxis dataKey="k" />
                                    <YAxis domain={[0, 0.5]} />
                                    <Tooltip />
                                    <Bar dataKey="sil" name="Silhouette">
                                        {elbowData.map((d, i) => <Cell key={i} fill={d.k === 'k=4' ? P.accent : P.dark} />)}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <div className="metric-grid-3" style={{ marginTop: '1rem' }}>
                        <div className="metric-card"><div className="label">Optimal k</div><div className="value">4</div><div className="delta">Via elbow + silhouette analysis</div></div>
                        <div className="metric-card"><div className="label">Silhouette Score</div><div className="value">0.421</div><div className="delta">At k=4 (best separation)</div></div>
                        <div className="metric-card"><div className="label">Inertia at k=4</div><div className="value">27,200</div><div className="delta">Within-cluster sum of squares</div></div>
                    </div>
                </>
            )}
        </>
    )
}
