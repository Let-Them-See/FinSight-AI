"""
FinSight AI — Synthetic Dataset Generator
==========================================
Generates four enterprise-grade, realistic synthetic datasets for
the FinSight AI platform modules. All data follows Indian BFSI
conventions: ₹ in Lakhs/Crores, pan-India geography, and realistic
financial distributions calibrated to HDFC/ICICI/Bajaj Finserv contexts.

Run this script ONCE before opening any notebooks:
    python generate_data.py

Author  : FinSight AI Team
Version : 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Reproducible seed ────────────────────────────────────────────────────────
np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("\n" + "="*65)
print("  FinSight AI — Synthetic Data Generator")
print("  Indian BFSI Context | Enterprise Grade")
print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Gujarat",
    "Rajasthan", "Uttar Pradesh", "West Bengal", "Delhi NCT", "Madhya Pradesh",
    "Punjab", "Haryana", "Kerala", "Andhra Pradesh", "Bihar",
    "Odisha", "Jharkhand", "Chhattisgarh", "Assam", "Himachal Pradesh"
]

GEO_TIERS = {
    "Tier-1": ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi NCT", "Gujarat",
               "Telangana", "West Bengal"],
    "Tier-2": ["Rajasthan", "Punjab", "Haryana", "Kerala", "Andhra Pradesh", "Madhya Pradesh"],
    "Tier-3": ["Uttar Pradesh", "Bihar", "Odisha", "Jharkhand", "Chhattisgarh",
               "Assam", "Himachal Pradesh"],
}
STATE_TO_TIER = {s: t for t, states in GEO_TIERS.items() for s in states}


def assign_credit_grade(credit_score: float) -> str:
    """Map CIBIL-style credit score to credit grade label."""
    if credit_score >= 800:
        return "AAA"
    elif credit_score >= 750:
        return "AA"
    elif credit_score >= 700:
        return "A"
    elif credit_score >= 650:
        return "BBB"
    elif credit_score >= 600:
        return "BB"
    else:
        return "B"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk Dataset (15,000 rows)
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_risk_data(n_samples: int = 15000) -> pd.DataFrame:
    """
    Generate a realistic loan applicant dataset for credit default prediction.

    Distributions calibrated to:
      - RBI MSME/Retail loan portfolio statistics (FY 2023)
      - CIBIL TransUnion India credit score distribution
      - HDFC Bank / Bajaj Finserv product mix

    Features include demographics, income, loan details, credit history,
    and geographic data — all in Indian enterprise conventions.
    """
    print(f"\n[Module 1] Generating Credit Risk Dataset ({n_samples:,} rows)...")

    # ── Demographics ──────────────────────────────────────────────────────────
    applicant_age    = np.random.randint(22, 65, n_samples)
    applicant_gender = np.random.choice(["Male", "Female", "Other"],
                                        n_samples, p=[0.62, 0.36, 0.02])

    # ── Employment and income (correlated with age) ───────────────────────────
    emp_type_weights  = [0.55, 0.25, 0.12, 0.08]  # Salaried, Self-Employed, Business, Contract
    employment_type   = np.random.choice(
        ["Salaried", "Self-Employed", "Business Owner", "Contract"],
        n_samples, p=emp_type_weights
    )

    # Annual income in Lakhs — log-normal for realistic right skew
    base_income = np.exp(np.random.normal(2.2, 0.55, n_samples))  # median ~₹9L
    income_boost = np.where(employment_type == "Business Owner", 1.6,
               np.where(employment_type == "Self-Employed", 1.2, 1.0))
    annual_income_lakh = np.clip(base_income * income_boost, 2.5, 180.0).round(2)

    employment_years = np.clip(np.random.exponential(5, n_samples), 0.5, 35.0).round(1)

    # ── Loan parameters ───────────────────────────────────────────────────────
    loan_purpose    = np.random.choice(
        ["Home Loan", "Personal Loan", "Auto Loan", "Business Loan", "Education Loan", "Gold Loan"],
        n_samples, p=[0.30, 0.28, 0.18, 0.12, 0.08, 0.04]
    )

    # Loan amount as function of income and purpose
    loan_multiplier = np.where(loan_purpose == "Home Loan", np.random.uniform(8, 20, n_samples),
                  np.where(loan_purpose == "Business Loan", np.random.uniform(5, 15, n_samples),
                  np.where(loan_purpose == "Auto Loan",     np.random.uniform(2, 8, n_samples),
                                                            np.random.uniform(0.5, 5, n_samples))))
    loan_amount_lakh = np.clip((annual_income_lakh * loan_multiplier) / 10, 0.5, 250.0).round(2)

    loan_tenure_months    = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240],
                                             n_samples, p=[0.05,0.10,0.18,0.15,0.20,0.13,0.10,0.05,0.04])
    interest_rate_pct     = np.clip(np.random.normal(10.5, 2.5, n_samples), 7.0, 20.0).round(2)

    # ── Credit profile ────────────────────────────────────────────────────────
    # CIBIL score: normal distribution centred at 700 (Indian market distribution)
    credit_score             = np.clip(np.random.normal(700, 75, n_samples), 300, 900).astype(int)
    credit_grade             = [assign_credit_grade(s) for s in credit_score]

    num_credit_accounts      = np.random.randint(1, 10, n_samples)
    existing_emis            = np.random.randint(0, 6, n_samples)
    months_since_last_default = np.where(
        np.random.rand(n_samples) < 0.35,  # 35% have prior defaults
        np.random.randint(1, 120, n_samples),
        999  # 999 = "no prior default" convention
    )

    # Debt-to-income ratio
    existing_debt_approx = existing_emis * annual_income_lakh * 0.12
    debt_to_income_ratio = np.clip(existing_debt_approx / annual_income_lakh, 0.0, 2.5).round(3)

    # ── Assets ────────────────────────────────────────────────────────────────
    total_assets_lakh = np.clip(
        annual_income_lakh * np.random.uniform(0.5, 8, n_samples) + np.random.uniform(0, 20, n_samples),
        0, 5000
    ).round(2)
    property_ownership = np.random.choice(
        ["Own", "Rented", "Parental", "Company Provided"],
        n_samples, p=[0.45, 0.35, 0.15, 0.05]
    )

    # ── Geography ─────────────────────────────────────────────────────────────
    _state_weights = np.array([12,9,9,8,8,6,7,6,7,5,3,3,4,4,4,2,2,2,2,1], dtype=float)
    _state_probs   = _state_weights / _state_weights.sum()
    state_name     = np.random.choice(INDIAN_STATES, n_samples, p=_state_probs)
    geographic_tier  = [STATE_TO_TIER[s] for s in state_name]

    # ── Target variable: Default Flag ─────────────────────────────────────────
    # Probability of default: logistic function of key risk drivers
    default_logit = (
        -3.5
        + 0.03  * (700 - credit_score)           # Low credit score = higher risk
        + 0.25  * debt_to_income_ratio            # High DTI = higher risk
        + 0.15  * (loan_amount_lakh / annual_income_lakh)  # High LTI = higher risk
        - 0.08  * (applicant_age - 30).clip(0)   # Age as protective factor
        + 0.30  * (months_since_last_default < 24).astype(int)  # Recent default
        + 0.20  * (employment_type == "Contract").astype(int)
        - 0.15  * (employment_type == "Salaried").astype(int)
        + np.random.normal(0, 0.5, n_samples)    # Idiosyncratic noise
    )
    default_probability = 1 / (1 + np.exp(-default_logit))
    default_flag        = np.where(
        np.random.rand(n_samples) < default_probability,
        "Yes", "No"
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "applicant_id"             : [f"APP{str(i).zfill(6)}" for i in range(1, n_samples + 1)],
        "applicant_age"            : applicant_age,
        "applicant_gender"         : applicant_gender,
        "employment_type"          : employment_type,
        "employment_years"         : employment_years,
        "annual_income_lakh"       : annual_income_lakh,
        "loan_purpose"             : loan_purpose,
        "loan_amount_lakh"         : loan_amount_lakh,
        "loan_tenure_months"       : loan_tenure_months,
        "interest_rate_pct"        : interest_rate_pct,
        "credit_score"             : credit_score,
        "credit_grade"             : credit_grade,
        "num_credit_accounts"      : num_credit_accounts,
        "existing_emis"            : existing_emis,
        "months_since_last_default": months_since_last_default,
        "debt_to_income_ratio"     : debt_to_income_ratio,
        "total_assets_lakh"        : total_assets_lakh,
        "property_ownership"       : property_ownership,
        "state_name"               : state_name,
        "geographic_tier"          : geographic_tier,
        "default_flag"             : default_flag,
    })

    # ── Introduce ~3% realistic missing values ────────────────────────────────
    for col in ["employment_years", "months_since_last_default", "total_assets_lakh"]:
        miss_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[miss_idx, col] = np.nan

    default_rate = (df["default_flag"] == "Yes").mean()
    print(f"[Module 1] ✅ Generated {n_samples:,} rows | Default rate: {default_rate:.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation Dataset (15,000 rows)
# ─────────────────────────────────────────────────────────────────────────────

def generate_customer_segmentation_data(n_samples: int = 15000) -> pd.DataFrame:
    """
    Generate a customer financial profile dataset for segmentation analysis.
    Modelled on Indian retail banking customer cohorts (HDFC/SBI/ICICI mix).
    """
    print(f"\n[Module 2] Generating Customer Segmentation Dataset ({n_samples:,} rows)...")

    # ── Age distribution across three life-stage cohorts ──────────────────────
    customer_age = np.concatenate([
        np.random.randint(22, 35, int(n_samples * 0.38)),  # Young professionals
        np.random.randint(35, 52, int(n_samples * 0.40)),  # Mid-career
        np.random.randint(52, 75, n_samples - int(n_samples * 0.78)),  # Senior
    ])
    np.random.shuffle(customer_age)

    gender               = np.random.choice(["Male", "Female", "Other"], n_samples, p=[0.60, 0.38, 0.02])
    occupation_category  = np.random.choice(
        ["IT/Tech", "Banking/Finance", "Government", "Healthcare",
         "Manufacturing", "Business Owner", "Retired", "Other"],
        n_samples, p=[0.18, 0.12, 0.15, 0.08, 0.10, 0.15, 0.12, 0.10]
    )

    # ── Income based on occupation ────────────────────────────────────────────
    income_map = {
        "IT/Tech": (12, 3.5), "Banking/Finance": (10, 3), "Government": (7, 2),
        "Healthcare": (9, 3), "Manufacturing": (6, 2), "Business Owner": (18, 7),
        "Retired": (4, 1.5), "Other": (5, 2)
    }
    monthly_income_lakh = np.array([
        np.clip(np.random.normal(income_map[occ][0], income_map[occ][1]) / 12, 0.3, 25)
        for occ in occupation_category
    ]).round(3)

    # ── Expenses, savings, investments, debt ──────────────────────────────────
    monthly_expenses_lakh   = np.clip(monthly_income_lakh * np.random.uniform(0.4, 0.85, n_samples), 0.1, 20).round(3)
    total_savings_lakh      = np.clip(monthly_income_lakh * 12 * np.random.uniform(0.5, 8, n_samples), 0.1, 500).round(2)
    total_investments_lakh  = np.clip(total_savings_lakh * np.random.uniform(0, 1.5, n_samples), 0, 1000).round(2)
    total_debt_lakh         = np.clip(monthly_income_lakh * 12 * np.random.uniform(0, 5, n_samples), 0, 800).round(2)

    # ── Banking and digital behaviour ─────────────────────────────────────────
    primary_bank             = np.random.choice(
        ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Mahindra", "PNB", "Bank of Baroda", "Other"],
        n_samples, p=[0.20, 0.22, 0.18, 0.12, 0.09, 0.07, 0.06, 0.06]
    )
    num_products_held        = np.random.randint(1, 9, n_samples)
    account_tenure_months    = np.random.randint(3, 300, n_samples)
    avg_monthly_transaction_count = np.random.randint(5, 120, n_samples)
    digital_engagement_score = np.clip(np.random.beta(3, 2, n_samples) * 100, 0, 100).round(1)
    credit_utilization_pct   = np.clip(np.random.beta(2, 3, n_samples) * 100, 0, 100).round(1)

    # ── Investment preference ─────────────────────────────────────────────────
    investment_preference = np.random.choice(
        ["Equity", "Mutual Funds", "Fixed Deposits", "Real Estate", "Gold", "Hybrid", "None"],
        n_samples, p=[0.15, 0.22, 0.25, 0.12, 0.10, 0.10, 0.06]
    )

    risk_appetite     = np.random.choice(["Low", "Moderate", "High"], n_samples, p=[0.35, 0.45, 0.20])
    customer_city_tier = np.random.choice(["Tier-1", "Tier-2", "Tier-3"], n_samples, p=[0.38, 0.37, 0.25])

    df = pd.DataFrame({
        "customer_id"                  : [f"CUST{str(i).zfill(7)}" for i in range(1, n_samples + 1)],
        "customer_age"                 : customer_age,
        "gender"                       : gender,
        "occupation_category"          : occupation_category,
        "monthly_income_lakh"          : monthly_income_lakh,
        "monthly_expenses_lakh"        : monthly_expenses_lakh,
        "total_savings_lakh"           : total_savings_lakh,
        "total_investments_lakh"       : total_investments_lakh,
        "total_debt_lakh"              : total_debt_lakh,
        "primary_bank"                 : primary_bank,
        "num_products_held"            : num_products_held,
        "account_tenure_months"        : account_tenure_months,
        "avg_monthly_transaction_count": avg_monthly_transaction_count,
        "digital_engagement_score"     : digital_engagement_score,
        "credit_utilization_pct"       : credit_utilization_pct,
        "investment_preference"        : investment_preference,
        "risk_appetite"                : risk_appetite,
        "customer_city_tier"           : customer_city_tier,
    })

    print(f"[Module 2] ✅ Generated {n_samples:,} rows | {df.shape[1]} features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Financial News Dataset (5,000 rows)
# ─────────────────────────────────────────────────────────────────────────────

def generate_financial_news_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic Indian financial news headlines with sentiment labels.
    Calibrated to Economic Times / Business Standard / Mint news patterns.
    """
    print(f"\n[Module 3] Generating Financial News Dataset ({n_samples:,} rows)...")

    POSITIVE_TEMPLATES = [
        "RBI keeps repo rate unchanged, signals accommodative stance for Q{q} FY{yr}",
        "{bank} Q{q} net profit surges {pct}%, beats Bloomberg analyst estimates",
        "Sensex rallies {pts} points as FIIs pump ₹{amt} Cr into Indian equities",
        "India GDP growth accelerates to {pct}% in Q{q}, ahead of IMF projections",
        "{company} announces ₹{amt} Cr CAPEX for pan-India digital infrastructure rollout",
        "SEBI approves simplified IPO norms; Dalal Street cheers listing reforms",
        "UPI transaction volume crosses {amt} Cr in {month}, new all-time high",
        "{sector} sector EBITDA margins expand QoQ amid easing input cost pressures",
        "India forex reserves hit {amt} billion USD as RBI intervention stabilises rupee",
        "{bank} upgrades credit outlook to 'Stable'; Moody's cites strong capital buffers",
        "Nifty 50 hits record {pts} as broad-based rally lifts mid-cap, small-cap indices",
        "GST collections cross ₹{amt} Cr for {month}th consecutive month",
        "IT majors TCS, Infosys report strong deal wins; sector outlook upgraded to 'Buy'",
    ]

    NEGATIVE_TEMPLATES = [
        "RBI hikes repo rate by {bps} bps amid persistent {reason} inflation concerns",
        "{bank} NPA ratio rises to {pct}% in Q{q}; provisions surge QoQ",
        "FII outflows drag Sensex {pts} points lower; rupee slips to {level} per USD",
        "India current account deficit widens to {pct}% of GDP in Q{q} FY{yr}",
        "{company} profit warning: EBITDA margins to decline {pct}% on high input costs",
        "SEBI probes {company} for insider trading; shares locked in lower circuit",
        "Credit card defaults surge {pct}% YoY; RBI flags retail lending risks",
        "Monsoon deficit at {pct}% below normal; agri-loan stress feared in Tier-2/3 markets",
        "{sector} sector faces GST scrutiny; ₹{amt} Cr ITC claims flagged by CBIC",
        "Bond yields spike {bps}bps after higher-than-expected US Fed rate guidance",
        "Rupee depreciates to historic low of ₹{level} on trade deficit widening",
        "{bank} board rejects merger proposal; institutional investors express concern",
    ]

    NEUTRAL_TEMPLATES = [
        "SEBI board meets to review derivative market norms for Q{q} FY{yr}",
        "RBI monetary policy committee holds repo rate at {rate}%; next review in {month}",
        "{bank} announces rights issue at ₹{price} per share; record date set",
        "India WPI inflation prints at {pct}% for {month}; IIP data due next week",
        "NCLAT hears {company} insolvency appeal; next hearing scheduled for {month}",
        "NSE updates margin framework for equity derivatives from {month} onwards",
        "Finance Ministry reviews PLI scheme performance across {n} key sectors",
        "{sector} sector FDI inflows at {amt} million USD in April-{month} FY{yr}",
        "RBI releases draft guidelines on digital lending regulations for public comment",
        "MSCI rebalancing: India's weight adjusted to {pct}%; {n} stocks added/removed",
    ]

    BANKS    = ["HDFC Bank", "ICICI Bank", "Axis Bank", "SBI", "Kotak Mahindra", "IndusInd Bank",
                "Yes Bank", "Bank of Baroda", "Punjab National Bank", "IDFC First Bank"]
    COMPANIES = ["Reliance Industries", "TCS", "Infosys", "Wipro", "Adani Enterprises",
                 "Bajaj Finance", "Zomato", "ONGC", "ITC", "Maruti Suzuki", "HDFC Life",
                 "Paytm", "Nykaa", "Delhivery"]
    SECTORS  = ["BFSI", "IT Services", "FMCG", "Auto", "Pharma", "Infrastructure",
                "Real Estate", "Energy", "Telecom", "Metals & Mining"]
    MONTHS   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    NEWS_SRC = ["Economic Times", "Business Standard", "Mint", "Financial Express",
                "MoneyControl", "BloombergQuint", "Reuters India", "CNBC TV18"]

    headlines, sentiment_labels, sectors_list, news_sources, pub_dates, market_impacts = [], [], [], [], [], []

    # Date range: Jan 2022 – Dec 2024
    date_range = pd.date_range("2022-01-01", "2024-12-31", periods=n_samples)

    for i in range(n_samples):
        yr  = np.random.randint(22, 25)
        q   = np.random.randint(1, 5)
        pct = round(np.random.uniform(3, 48), 1)
        pts = np.random.randint(100, 1200)
        amt = np.random.randint(200, 5000)
        bps = np.random.choice([25, 35, 50])
        rate = round(np.random.uniform(6.0, 7.5), 2)
        price = np.random.randint(50, 500)
        level = round(np.random.uniform(82, 87), 2)
        n_items = np.random.randint(2, 15)
        month_name = np.random.choice(MONTHS)
        reason = np.random.choice(["food", "core", "energy", "services"])
        bank = np.random.choice(BANKS)
        company = np.random.choice(COMPANIES)
        sector = np.random.choice(SECTORS)

        rnd = np.random.rand()
        if rnd < 0.38:
            tpl = np.random.choice(POSITIVE_TEMPLATES)
            sentiment = "Positive"
            impact = "Bullish"
        elif rnd < 0.65:
            tpl = np.random.choice(NEGATIVE_TEMPLATES)
            sentiment = "Negative"
            impact = "Bearish"
        else:
            tpl = np.random.choice(NEUTRAL_TEMPLATES)
            sentiment = "Neutral"
            impact = "Neutral"

        headline = tpl.format(
            bank=bank, company=company, sector=sector,
            q=q, yr=yr, pct=pct, pts=pts, amt=amt, bps=bps,
            rate=rate, price=price, level=level, n=n_items,
            month=month_name, reason=reason
        )

        headlines.append(headline)
        sentiment_labels.append(sentiment)
        sectors_list.append(sector)
        news_sources.append(np.random.choice(NEWS_SRC))
        pub_dates.append(date_range[i])
        market_impacts.append(impact)

    df = pd.DataFrame({
        "headline"        : headlines,
        "publication_date": pub_dates,
        "news_source"     : news_sources,
        "sector"          : sectors_list,
        "sentiment_label" : sentiment_labels,
        "market_impact"   : market_impacts,
    })

    print(f"[Module 3] ✅ Generated {n_samples:,} rows | Sentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string(index=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series Dataset (monthly, FY2014–FY2024 = 120 months)
# ─────────────────────────────────────────────────────────────────────────────

def generate_time_series_data() -> pd.DataFrame:
    """
    Generate a 10-year monthly financial time series for an Indian NBFC/bank.
    Metrics include loan disbursements, NPAs, UPI volumes, and revenue —
    all in ₹ Crores with realistic trends, seasonality, and shock events.
    """
    print(f"\n[Module 4] Generating Time Series Dataset (Monthly, FY2014–FY2024)...")

    months = pd.date_range(start="2014-04-01", end="2024-03-01", freq="MS")
    n      = len(months)

    # ── Trend component: compound growth ~12% p.a. ────────────────────────────
    trend_growth = np.array([1.12 ** (t / 12) for t in range(n)])

    # ── Indian fiscal seasonality: Q4 (Jan-Mar) surge, Q1 (Apr-Jun) dip ──────
    fiscal_seasonal = np.array([
        1.05 if m.month in [1, 2, 3]      else   # Q4 FY — year-end push
        0.92 if m.month in [4, 5]         else   # Q1 start — slow
        1.03 if m.month in [10, 11]       else   # Festival season (Diwali/Navratri)
        1.00
        for m in months
    ])

    # ── COVID-19 shock: Mar 2020–Sep 2020 ────────────────────────────────────
    covid_shock = np.array([
        0.55 if pd.Timestamp("2020-03-01") <= m <= pd.Timestamp("2020-09-01") else
        0.80 if pd.Timestamp("2020-10-01") <= m <= pd.Timestamp("2021-03-01") else
        1.00
        for m in months
    ])

    # ── Base disbursement: ₹1,200 Cr in April 2014 ───────────────────────────
    base_value  = 1200.0
    noise       = np.random.normal(1.0, 0.04, n)

    total_loan_disbursement_cr = (base_value * trend_growth * fiscal_seasonal * covid_shock * noise).round(2)

    # ── NPA rate: rises during stress, improves post-COVID ───────────────────
    base_npa    = 4.2
    npa_trend   = np.linspace(0, 2.5, n)
    covid_npa   = np.array([
        2.5 if pd.Timestamp("2020-03-01") <= m <= pd.Timestamp("2021-06-01") else
        1.0 if pd.Timestamp("2021-07-01") <= m <= pd.Timestamp("2022-06-01") else
        -0.5 if m >= pd.Timestamp("2022-07-01") else 0
        for m in months
    ])
    npa_rate_pct = np.clip(base_npa + npa_trend * 0.05 + covid_npa +
                           np.random.normal(0, 0.2, n), 1.5, 12.0).round(2)

    # ── Revenue (correlated with disbursements) ───────────────────────────────
    interest_income_cr    = (total_loan_disbursement_cr * np.random.uniform(0.09, 0.12, n) / 12).round(2)
    fee_income_cr         = (total_loan_disbursement_cr * np.random.uniform(0.005, 0.015, n)).round(2)
    total_revenue_cr      = (interest_income_cr + fee_income_cr).round(2)

    # ── UPI transactions: exponential growth from 2016 ───────────────────────
    upi_start = np.where(months >= pd.Timestamp("2016-08-01"), 1, 0)
    upi_months_since_launch = np.cumsum(upi_start)
    upi_transaction_cr      = (upi_months_since_launch ** 2.1 * 0.08 *
                                np.random.uniform(0.95, 1.05, n) * fiscal_seasonal).round(2)
    upi_transaction_cr      = np.where(upi_start == 0, 0, upi_transaction_cr)

    # ── CASA ratio ────────────────────────────────────────────────────────────
    casa_ratio_pct = np.clip(np.random.normal(42, 4, n), 30, 60).round(1)

    # ── Capital adequacy ratio (CRAR) ─────────────────────────────────────────
    crar_pct       = np.clip(np.random.normal(15.5, 1.5, n), 11.0, 22.0).round(2)

    df = pd.DataFrame({
        "month"                       : months,
        "total_loan_disbursement_cr"  : total_loan_disbursement_cr,
        "npa_rate_pct"                 : npa_rate_pct,
        "interest_income_cr"           : interest_income_cr,
        "fee_income_cr"                : fee_income_cr,
        "total_revenue_cr"             : total_revenue_cr,
        "upi_transaction_cr"           : upi_transaction_cr,
        "casa_ratio_pct"               : casa_ratio_pct,
        "crar_pct"                     : crar_pct,
    })

    print(f"[Module 4] ✅ Generated {len(df)} monthly records | "
          f"Range: {df['month'].min().strftime('%b %Y')} → {df['month'].max().strftime('%b %Y')}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Generate all datasets and save to /data
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = {
        "credit_risk_data.csv"       : generate_credit_risk_data(15000),
        "customer_segments_data.csv" : generate_customer_segmentation_data(15000),
        "financial_news_data.csv"    : generate_financial_news_data(5000),
        "time_series_data.csv"       : generate_time_series_data(),
    }

    print("\n" + "-"*65)
    print("  Saving datasets to ./data/")
    print("-"*65)

    for filename, df in datasets.items():
        save_path = DATA_DIR / filename
        df.to_csv(save_path, index=False)
        print(f"  ✅ Saved: {filename:40s} ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    print("\n" + "="*65)
    print("  All datasets generated successfully!")
    print(f"  Location: {DATA_DIR}")
    print("="*65)
    print("\nNext step: Open any notebook in ./notebooks/ to begin analysis.")
