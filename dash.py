import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Tida Sports Academy Analytics", layout="wide", page_icon="⚽")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.9rem;
        font-weight: 600;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .kpi-subtitle {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)
# ================== DATABASE CONFIG ==================
import os

try:
    # Try Streamlit secrets first (for local development)
    if "database" in st.secrets:
        DB_USER = st.secrets["database"]["DB_USER"]
        DB_PASS = st.secrets["database"]["DB_PASS"]
        DB_HOST = st.secrets["database"]["DB_HOST"]
        DB_PORT = str(st.secrets["database"]["DB_PORT"])
        DB_NAME = st.secrets["database"]["DB_NAME"]
        TABLE_NAME = st.secrets["database"]["TABLE_NAME"]
        st.success("✅ Using Streamlit secrets")
    else:
        raise KeyError("No secrets found")
        
except (KeyError, FileNotFoundError):
    # Use environment variables (for Railway deployment)
    DB_USER = os.getenv("DB_USER", "streamlit_user")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = str(os.getenv("DB_PORT", "58323"))
    DB_NAME = os.getenv("DB_NAME", "railway")
    TABLE_NAME = os.getenv("TABLE_NAME", "data_final_slim")
    st.success("✅ Using environment variables")

# ADD THIS DEBUG LINE
#st.write(f"🔍 DEBUG - DB_USER being used: **{DB_USER}**")

# Validate credentials
if not all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME, TABLE_NAME]):
    st.error("❌ Missing database credentials!")
    st.stop()
    
st.success(f"✅ Loaded secrets - Connecting to: {DB_HOST}:{DB_PORT}/{DB_NAME}")

@st.cache_resource
def get_db_engine():
    """Create database connection engine"""
    ENGINE_URL = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    st.write(f"🔗 Connection string: mysql+pymysql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return create_engine(ENGINE_URL)

# ================== HELPER FUNCTIONS ==================
def parse_billing_info(billing_period, billing_interval):
    """Calculate package duration in months from billing period and interval."""
    if pd.isna(billing_period) or pd.isna(billing_interval):
        return 1
    period = str(billing_period).lower()
    try:
        interval = int(float(billing_interval))
    except Exception:
        interval = 1
    if 'year' in period:
        return interval * 12
    elif 'month' in period:
        return max(1, interval)
    elif 'week' in period:
        return max(1, int(round(interval * 7 / 30)))
    elif 'day' in period:
        return max(1, int(round(interval / 30)))
    else:
        return 1

def is_multi_sport(sport_string):
    """Check if customer has opted for MULTI SPORT (NOT gold)."""
    if pd.isna(sport_string) or sport_string == '':
        return False
    s_lower = str(sport_string).lower()
    if 'gold' in s_lower:
        return False
    parts = re.split(r'[+,&/|]', str(sport_string))
    parts = [p.strip() for p in parts if p.strip()]
    if any(keyword in s_lower for keyword in ['multi sport', 'all sport', 'unlimited']):
        return True
    return len(parts) > 1

def count_sports(sport_string):
    """Count number of sports in a string."""
    if pd.isna(sport_string) or sport_string == '':
        return 0
    s_lower = str(sport_string).lower()
    if 'gold' in s_lower:
        return 99
    parts = re.split(r'[+,&/|]', str(sport_string))
    return len([p for p in parts if p.strip()])

def extract_primary_sport(sport_string):
    """Extract the first/primary sport from a string."""
    if pd.isna(sport_string) or sport_string == '':
        return 'Unknown'
    s_lower = str(sport_string).lower()
    if 'gold' in s_lower:
        return 'Gold Membership'
    if 'multi sport' in s_lower:
        return 'Multi Sport'
    parts = re.split(r'[+,&/|]', str(sport_string))
    first = parts[0].strip() if parts else 'Unknown'
    return first.title() if first else 'Unknown'

def calculate_renewal_date(start_date, next_payment_date, package_duration):
    """Calculate renewal date."""
    if pd.notna(next_payment_date):
        return next_payment_date
    elif pd.notna(start_date) and pd.notna(package_duration):
        return start_date + pd.DateOffset(months=int(package_duration))
    else:
        return pd.NaT

def calculate_retention_risk(days):
    """Calculate retention risk score."""
    if pd.isna(days):
        return 'Unknown'
    d = int(days)
    if d < 0:
        return 'Critical'
    elif d <= 14:
        return 'High'
    else:
        return 'Active'

def clean_phone_number(num):
    """Keep only digits, normalise Indian mobile numbers."""
    if pd.isna(num) or str(num).strip() == "":
        return ""
    s = re.sub(r"\D", "", str(num))
    if len(s) == 10:
        s = "91" + s
    return s

def prepare_sports_data(df):
    """Prepare detailed sports breakdown from 'sport' column."""
    sports_data = []
    for _, row in df.iterrows():
        sport_str = row.get('sport', '')
        if pd.isna(sport_str) or str(sport_str).strip() == '':
            continue
        s_lower = str(sport_str).lower()
        base_revenue = row.get('product_net_revenue', 0) or 0
        base_mfee = row.get('Monthly Equivalent Fee', 0) or 0
        school = row.get('school', '')
        place = row.get('place', '')
        
        if 'gold' in s_lower:
            sports_data.append({
                'Sport': 'Gold Membership',
                'Revenue': base_revenue,
                'Monthly Fee': base_mfee,
                'School': school,
                'Place': place,
                'Type': 'Gold Membership'
            })
            continue
        
        parts = re.split(r'[+,&/|]', str(sport_str))
        parts = [p.strip() for p in parts if p.strip()]
        if 'multi sport' in s_lower or 'all sport' in s_lower or len(parts) > 1:
            sports_data.append({
                'Sport': 'Multi Sport',
                'Revenue': base_revenue,
                'Monthly Fee': base_mfee,
                'School': school,
                'Place': place,
                'Type': 'Multi Sport'
            })
            continue
        
        label = parts[0].title() if parts else str(sport_str).strip().title()
        sports_data.append({
            'Sport': label,
            'Revenue': base_revenue,
            'Monthly Fee': base_mfee,
            'School': school,
            'Place': place,
            'Type': 'Single Sport'
        })
    
    if not sports_data:
        return pd.DataFrame(columns=['Sport', 'Revenue', 'Monthly Fee', 'School', 'Place', 'Type'])
    return pd.DataFrame(sports_data)

def fix_bar_label_clipping(fig):
    """Prevent labels from being cut off by setting cliponaxis to False and adjusting margins."""
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        margin=dict(t=60, b=120, l=80, r=40),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )
    return fig

# ================== LOAD DATA FROM DB ==================
@st.cache_data(ttl=300)
def load_data_from_db():
    """Load data from MySQL database"""
    try:
        engine = get_db_engine()
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, engine)
        df.columns = df.columns.str.strip()
        
        date_cols = ['start_date', 'next_payment_date', 'wc_order_date_created']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if 'product_net_revenue' in df.columns:
            df['product_net_revenue'] = pd.to_numeric(df['product_net_revenue'], errors='coerce').fillna(0.0)
        else:
            df['product_net_revenue'] = 0.0
        
        if 'coupon_amount' in df.columns:
            df['coupon_amount'] = pd.to_numeric(df['coupon_amount'], errors='coerce').fillna(0.0)
        else:
            df['coupon_amount'] = 0.0
        
        if 'billing_period' in df.columns and 'billing_interval' in df.columns:
            df['Package Duration (Months)'] = df.apply(
                lambda row: parse_billing_info(row['billing_period'], row['billing_interval']),
                axis=1
            )
        else:
            df['Package Duration (Months)'] = 1
        
        df['Renewal Date'] = df.apply(
            lambda row: calculate_renewal_date(
                row.get('start_date'),
                row.get('next_payment_date'),
                row['Package Duration (Months)']
            ),
            axis=1
        )
        
        if 'sport' in df.columns:
            df['Is Multi-Sport'] = df['sport'].apply(is_multi_sport)
            df['Sport Count'] = df['sport'].apply(count_sports)
            df['Primary Sport'] = df['sport'].apply(extract_primary_sport)
            df['Is Gold'] = df['sport'].str.contains('gold', case=False, na=False)
            
            def membership_type(s):
                s_lower = str(s).lower()
                if 'gold' in s_lower:
                    return 'Gold Membership'
                elif is_multi_sport(s):
                    return 'Multi Sport'
                else:
                    return 'Single Sport'
            
            df['Membership Type'] = df['sport'].apply(membership_type)
        else:
            df['Is Multi-Sport'] = False
            df['Sport Count'] = 0
            df['Primary Sport'] = ''
            df['Is Gold'] = False
            df['Membership Type'] = 'Single Sport'
        
        df['Monthly Equivalent Fee'] = (df['product_net_revenue'] / df['Package Duration (Months)']) \
            .replace([float('inf'), -float('inf')], 0).round(2).fillna(0.0)
        
        today = pd.Timestamp.now().normalize()
        df['Days Until Renewal'] = (df['Renewal Date'] - today).dt.days
        df['Retention Risk'] = df['Days Until Renewal'].apply(calculate_retention_risk)
        
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['Days Since Start'] = (today - df['start_date']).dt.days
        else:
            df['Days Since Start'] = pd.NA
        
        df['Payment Age Bucket'] = pd.cut(
            df['Days Since Start'],
            bins=[-1, 30, 90, 180, 365*20],
            labels=['0-30 days', '31-90 days', '91-180 days', '180+ days']
        )
        
        def calculate_months_overdue(row):
            if row.get('Retention Risk') == 'Critical':
                if pd.notna(row.get('Renewal Date')):
                    months_overdue = max(1, int((today - row['Renewal Date']).days / 30))
                    return months_overdue
            return 0
        
        df['Months Overdue'] = df.apply(calculate_months_overdue, axis=1)
        df['Due Amount'] = df.apply(
            lambda row: row['Monthly Equivalent Fee'] * row['Months Overdue']
            if row['Months Overdue'] > 0 else 0,
            axis=1
        )
        
        if 'phone' in df.columns:
            df['phone'] = df['phone'].apply(clean_phone_number)
        
        text_cols = [
            'wc_order_for_name', 'sport', 'school', 'place', 'academy_name', 'phone', 'email',
            'wc_order_attribution_device_type', 'wc_order_attribution_utm_source'
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        if 'product_qty' in df.columns:
            df['product_qty'] = pd.to_numeric(df['product_qty'], errors='coerce').fillna(0).astype(int)
        else:
            df['product_qty'] = 0
        
        if 'wc_order_for_name' in df.columns:
            df['__norm_name'] = df['wc_order_for_name'].astype(str).str.strip().str.lower()
            codes, uniques = pd.factorize(df['__norm_name'], sort=True)
            df['student_id'] = ['SID' + str(i + 1).zfill(6) for i in codes]
            first_map = df.groupby('student_id')['start_date'].min().to_dict()
            df['first_order_date'] = df['student_id'].map(first_map)
        else:
            df['student_id'] = ''
            df['first_order_date'] = pd.NaT
        
        return df
    except Exception as e:
        st.error(f"❌ Error connecting to database: {str(e)}")
        return None

# ================== MAIN APP ==================
st.title("⚽ Sports Academy Analytics Dashboard")
st.markdown("#### Revenue, Renewals & Retention Overview")

with st.spinner('🔄 Loading data from database...'):
    df = load_data_from_db()

if df is None or df.empty:
    st.error("❌ No data loaded from database")
    st.stop()

st.success(f"✅ Loaded {len(df):,} records from MySQL database")
st.caption(f"Dashboard generated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# ================== SIDEBAR FILTERS ==================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/filter.png", width=50)
st.sidebar.title("Filters")

date_filter = st.sidebar.selectbox(
    "Time Period (by Start Date)",
    ["All Time", "Last Week", "Last 2 Weeks", "Last Month", "Last Quarter", "Last Year", "Custom"]
)

if date_filter == "Custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        default_start = df['start_date'].min() if 'start_date' in df.columns else (datetime.now() - timedelta(days=365))
        start = st.date_input("From", value=default_start.date() if hasattr(default_start, "date") else default_start)
    with col2:
        end = st.date_input("To", value=datetime.now().date())
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered_df = df[(df['start_date'].notna()) & (df['start_date'] >= start_dt) & (df['start_date'] <= end_dt)]
else:
    days_map = {
        "Last Week": 7,
        "Last 2 Weeks": 14,
        "Last Month": 30,
        "Last Quarter": 90,
        "Last Year": 365
    }
    if date_filter in days_map:
        cutoff = datetime.now() - timedelta(days=days_map[date_filter])
        filtered_df = df[df['start_date'] >= cutoff]
    else:
        filtered_df = df.copy()

st.sidebar.markdown("---")

if 'sport' in df.columns:
    st.sidebar.markdown("### 🏃 Sport Filter")
    sport_values = sorted([s for s in df['sport'].unique() if s and str(s).strip() != ''])
    if sport_values:
        selected_sports = st.sidebar.multiselect(
            "Select Sports",
            ["All"] + sport_values,
            default=["All"]
        )
        if "All" not in selected_sports and selected_sports:
            filtered_df = filtered_df[filtered_df['sport'].isin(selected_sports)]

st.sidebar.markdown("---")

multi_sport_filter = st.sidebar.radio(
    "Membership Type",
    ["All", "Single Sport Only", "Multi-Sport Only", "Gold Membership Only"]
)

if multi_sport_filter == "Single Sport Only":
    filtered_df = filtered_df[(~filtered_df['Is Multi-Sport']) & (~filtered_df['Is Gold'])]
elif multi_sport_filter == "Multi-Sport Only":
    filtered_df = filtered_df[(filtered_df['Is Multi-Sport']) & (~filtered_df['Is Gold'])]
elif multi_sport_filter == "Gold Membership Only":
    filtered_df = filtered_df[filtered_df['Is Gold']]

if 'school' in df.columns:
    schools = sorted([s for s in df['school'].unique() if s and str(s).strip() != ''])
    if schools:
        selected_schools = st.sidebar.multiselect(
            "Schools",
            ["All"] + schools,
            default=["All"]
        )
        if "All" not in selected_schools:
            filtered_df = filtered_df[filtered_df['school'].isin(selected_schools)]

if 'place' in df.columns:
    places = sorted([p for p in df['place'].unique() if p and str(p).strip() != ''])
    if places:
        selected_places = st.sidebar.multiselect(
            "Locations",
            ["All"] + places,
            default=["All"]
        )
        if "All" not in selected_places:
            filtered_df = filtered_df[filtered_df['place'].isin(selected_places)]

if 'academy_name' in df.columns:
    academies = sorted([a for a in df['academy_name'].unique() if a and str(a).strip() != ''])
    if academies:
        selected_academies = st.sidebar.multiselect(
            "Academies",
            ["All"] + academies,
            default=["All"]
        )
        if "All" not in selected_academies:
            filtered_df = filtered_df[filtered_df['academy_name'].isin(selected_academies)]

if 'Retention Risk' in df.columns:
    st.sidebar.markdown("---")
    risk_levels = st.sidebar.multiselect(
        "Retention Status",
        ["All", "Active", "High", "Critical", "Unknown"],
        default=["All"]
    )
    if "All" not in risk_levels:
        filtered_df = filtered_df[filtered_df['Retention Risk'].isin(risk_levels)]

if 'wc_order_attribution_device_type' in df.columns:
    st.sidebar.markdown("---")
    device_types = sorted([d for d in df['wc_order_attribution_device_type'].unique() if d and str(d).strip() != ''])
    if device_types:
        selected_devices = st.sidebar.multiselect(
            "Device Type",
            ["All"] + device_types,
            default=["All"]
        )
        if "All" not in selected_devices:
            filtered_df = filtered_df[filtered_df['wc_order_attribution_device_type'].isin(selected_devices)]

if 'wc_order_attribution_utm_source' in df.columns:
    sources = sorted([s for s in df['wc_order_attribution_utm_source'].unique() if s and str(s).strip() != ''])
    if sources:
        selected_sources = st.sidebar.multiselect(
            "Traffic Source",
            ["All"] + sources,
            default=["All"]
        )
        if "All" not in selected_sources:
            filtered_df = filtered_df[filtered_df['wc_order_attribution_utm_source'].isin(selected_sources)]

st.sidebar.markdown("---")

with st.sidebar.expander("📘 Dashboard Guide"):
    st.markdown("""
**Key Concepts:**
- **MRR**: Monthly Recurring Revenue
- **Multi-Sport**: 'Multi Sport' or multiple sports
- **Gold Membership**: separate product
- **Retention Risk**: Based on days until renewal
- **Months Overdue**: How many months past renewal date

**Risk Levels:**
- 🔴 Critical: Overdue (negative days)
- 🟡 High: Due within 14 days
- 🟢 Active: More than 14 days away
""")

# ================== KEY METRICS (ALL KPIs FROM ORIGINAL) ==================
st.markdown('<div class="section-header">📊 Key Performance Indicators (KPIs)</div>', unsafe_allow_html=True)

# Calculate all KPI values
total_customers = len(filtered_df)
multi_sport_customers = len(filtered_df[filtered_df['Is Multi-Sport']])
multi_pct = (multi_sport_customers / total_customers * 100) if total_customers > 0 else 0
total_enrollments = filtered_df['product_qty'].sum() if 'product_qty' in filtered_df.columns else 0
total_revenue = filtered_df['product_net_revenue'].sum()
monthly_rev = filtered_df['Monthly Equivalent Fee'].sum()
overdue = filtered_df['Due Amount'].sum()
overdue_pct = (overdue / total_revenue * 100) if total_revenue > 0 else 0

critical_count = len(filtered_df[filtered_df['Retention Risk'] == 'Critical'])
high_count = len(filtered_df[filtered_df['Retention Risk'] == 'High'])
churn_risk = critical_count + high_count
churn_rate = (churn_risk / total_customers * 100) if total_customers > 0 else 0

multi_month = len(filtered_df[filtered_df['Package Duration (Months)'] > 1])
multi_month_pct = (multi_month / total_customers * 100) if total_customers > 0 else 0

upcoming_14 = len(filtered_df[filtered_df['Days Until Renewal'].between(0, 14, inclusive="both")])

num_academies = 0
if 'academy_name' in filtered_df.columns:
    num_academies = filtered_df['academy_name'].replace('', pd.NA).dropna().nunique()

card_pmt = 0
card_pct = 0
if 'wc_order_attribution_utm_source' in filtered_df.columns:
    card_pmt = len(filtered_df[filtered_df['wc_order_attribution_utm_source'].str.contains('card', case=False, na=False)])
    card_pct = (card_pmt / total_customers * 100) if total_customers > 0 else 0

# Coupon metrics
coupon_total = filtered_df['coupon_amount'].sum() if 'coupon_amount' in filtered_df.columns else 0
coupon_orders = (filtered_df['coupon_amount'] > 0).sum() if 'coupon_amount' in filtered_df.columns else 0
avg_coupon = coupon_total / coupon_orders if coupon_orders > 0 else 0

# FIRST ROW: PRIMARY KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
        <div class="kpi-subtitle">{multi_sport_customers} multi-sport ({multi_pct:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Enrollments</div>
        <div class="kpi-value">{total_enrollments:.0f}</div>
        <div class="kpi-subtitle">Product quantity sum</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Revenue Collected</div>
        <div class="kpi-value">₹{total_revenue:,.0f}</div>
        <div class="kpi-subtitle">Product net revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">MRR</div>
        <div class="kpi-value">₹{monthly_rev:,.0f}</div>
        <div class="kpi-subtitle">Monthly Recurring Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Outstanding Dues</div>
        <div class="kpi-value">₹{overdue:,.0f}</div>
        <div class="kpi-subtitle">{overdue_pct:.1f}% of collected</div>
    </div>
    """, unsafe_allow_html=True)

# SECOND ROW: SECONDARY KPIs
col6, col7, col8, col9, col10 = st.columns(5)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Customers Needing Follow-up</div>
        <div class="kpi-value">{churn_risk}</div>
        <div class="kpi-subtitle">{churn_rate:.1f}% of base</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Multi-Month Plans</div>
        <div class="kpi-value">{multi_month}</div>
        <div class="kpi-subtitle">{multi_month_pct:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Renewals Due in 14 Days</div>
        <div class="kpi-value">{upcoming_14}</div>
        <div class="kpi-subtitle">Upcoming renewals</div>
    </div>
    """, unsafe_allow_html=True)

with col9:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Active Academies</div>
        <div class="kpi-value">{num_academies}</div>
        <div class="kpi-subtitle">Unique academy count</div>
    </div>
    """, unsafe_allow_html=True)

with col10:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Card Payments</div>
        <div class="kpi-value">{card_pct:.1f}%</div>
        <div class="kpi-subtitle">{card_pmt} transactions</div>
    </div>
    """, unsafe_allow_html=True)

# THIRD ROW: COUPON METRICS
st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Coupon Discounts</div>
        <div class="kpi-value">₹{coupon_total:,.0f}</div>
        <div class="kpi-subtitle">Total discount amount</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Orders with Coupon</div>
        <div class="kpi-value">{coupon_orders}</div>
        <div class="kpi-subtitle">Coupon usage count</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Discount per Coupon Order</div>
        <div class="kpi-value">₹{avg_coupon:,.0f}</div>
        <div class="kpi-subtitle">Average coupon value</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Overdue alert (from original code)
total_overdue = filtered_df['Due Amount'].sum()
overdue_customers = len(filtered_df[filtered_df['Due Amount'] > 0])
if total_overdue > 0 and overdue_customers > 0:
    avg_months_overdue = filtered_df[filtered_df['Months Overdue'] > 0]['Months Overdue'].mean()
    st.error(f"""
⚠️ **COLLECTION ALERT**: {overdue_customers} customers have ₹{total_overdue:,.0f} in outstanding dues 
(Average: {avg_months_overdue:.1f} months overdue). Review Priority Follow-up List below!
""")

# ================== ENHANCED SPORTS ANALYSIS ==================
st.markdown("---")
st.markdown('<div class="metric-card">🎯 Comprehensive Sports Analysis</div>', unsafe_allow_html=True)
st.markdown("")

# Debug info for sport column
if 'sport' in filtered_df.columns:
    total_sport_entries = len(filtered_df)
    non_empty_sport = len(filtered_df[filtered_df['sport'].notna() & (filtered_df['sport'] != '')])

    if non_empty_sport == 0:
        st.warning(f"""
⚠️ **No sport data found**: Out of {total_sport_entries} records, {non_empty_sport} have sport information.

Please check:
- Is the 'sport' column populated in your database?
- Try removing filters to see if data appears
        """)
        with st.expander("🔍 View Sample Data (First 5 Records)"):

            sample_cols = ['wc_order_for_name', 'sport', 'product_net_revenue', 'start_date']
            available_cols = [c for c in sample_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[available_cols].head())
else:
    st.error("❌ 'sport' column not found in database table")
    st.stop()

sports_df = prepare_sports_data(filtered_df)

if not sports_df.empty and len(sports_df) > 0:
    # Top row: Sport Popularity & Revenue
    col1a, col2a = st.columns(2)

    with col1a:
        st.subheader("🏆 Top 10 Sports by Enrollments")
        sport_counts = sports_df['Sport'].value_counts().head(10).reset_index()
        sport_counts.columns = ['Sport', 'Enrollments']

        fig_sports = px.bar(
            sport_counts,
            x='Sport',
            y='Enrollments',
            text='Enrollments',
            color='Enrollments',
            color_continuous_scale='Viridis',
            title="Most Popular Sports"
        )
        fig_sports.update_traces(textposition='outside')
        fig_sports.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            height=450,
            xaxis_title="",
            yaxis_title="Number of Enrollments"
        )
        fix_bar_label_clipping(fig_sports)
        st.plotly_chart(fig_sports, use_container_width=True)

    with col2a:
        st.subheader("💰 Top 10 Sports by Revenue")
        sport_revenue = sports_df.groupby('Sport')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()

        fig_rev = px.bar(
            sport_revenue,
            x='Sport',
            y='Revenue',
            text='Revenue',
            color='Revenue',
            color_continuous_scale='Blues',
            title="Highest Revenue Generating Sports"
        )
        fig_rev.update_traces(
            texttemplate='₹%{text:,.0f}',
            textposition='outside'
        )
        fig_rev.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            height=450,
            xaxis_title="",
            yaxis_title="Total Revenue (₹)"
        )
        fix_bar_label_clipping(fig_rev)
        st.plotly_chart(fig_rev, use_container_width=True)

# Middle row: Single vs Multi-Sport vs Gold Comparison
st.markdown("---")
st.markdown('<div class="section-header">📊 Single, Multi Sport & Gold Membership Comparison</div>', unsafe_allow_html=True)
st.markdown("")

# Calculate membership metrics
membership_counts = filtered_df['Membership Type'].value_counts()
single_count = membership_counts.get('Single Sport', 0)
multi_count = membership_counts.get('Multi Sport', 0)
gold_count = membership_counts.get('Gold Membership', 0)

membership_revenue = filtered_df.groupby('Membership Type').agg({
    'product_net_revenue': 'sum',
    'Monthly Equivalent Fee': 'sum'
}).reset_index()

single_rev = membership_revenue[membership_revenue['Membership Type'] == 'Single Sport']['product_net_revenue'].sum() if 'Single Sport' in membership_revenue['Membership Type'].values else 0
multi_rev = membership_revenue[membership_revenue['Membership Type'] == 'Multi Sport']['product_net_revenue'].sum() if 'Multi Sport' in membership_revenue['Membership Type'].values else 0
gold_rev = membership_revenue[membership_revenue['Membership Type'] == 'Gold Membership']['product_net_revenue'].sum() if 'Gold Membership' in membership_revenue['Membership Type'].values else 0

single_mrr = membership_revenue[membership_revenue['Membership Type'] == 'Single Sport']['Monthly Equivalent Fee'].sum() if 'Single Sport' in membership_revenue['Membership Type'].values else 0
multi_mrr = membership_revenue[membership_revenue['Membership Type'] == 'Multi Sport']['Monthly Equivalent Fee'].sum() if 'Multi Sport' in membership_revenue['Membership Type'].values else 0
gold_mrr = membership_revenue[membership_revenue['Membership Type'] == 'Gold Membership']['Monthly Equivalent Fee'].sum() if 'Gold Membership' in membership_revenue['Membership Type'].values else 0

avg_single = single_rev / single_count if single_count > 0 else 0
avg_multi = multi_rev / multi_count if multi_count > 0 else 0
avg_gold = gold_rev / gold_count if gold_count > 0 else 0

# ROW 1: Membership Distribution (Pie) + Customer Counts (Bar)
col3a, col4a = st.columns([1, 1])

with col3a:
    st.markdown("##### 📊 Membership Distribution")
    membership_dist = filtered_df['Membership Type'].value_counts().reset_index()
    membership_dist.columns = ['Type', 'Count']

    fig_membership = px.pie(
        membership_dist,
        values='Count',
        names='Type',
        hole=0.5,
        color_discrete_sequence=['#fbbf24', '#3b82f6', '#10b981'],
        title=""
    )
    fig_membership.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14,
        pull=[0.05, 0.05, 0.05]
    )
    fig_membership.update_layout(
        height=400, 
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05,
            font=dict(size=12)
        )
    )
    st.plotly_chart(fig_membership, use_container_width=True)

with col4a:
    st.markdown("##### 👥 Customer Count by Membership Type")
    
    fig_cust_count = go.Figure()
    fig_cust_count.add_trace(go.Bar(
        x=membership_dist['Type'],
        y=membership_dist['Count'],
        text=membership_dist['Count'],
        textposition='outside',
        marker=dict(
            color=['#fbbf24', '#3b82f6', '#10b981'],
            line=dict(width=0)
        )
    ))
    fig_cust_count.update_layout(
        height=400,
        margin=dict(t=50, b=100, l=60, r=40),
        yaxis_title="Number of Customers",
        xaxis=dict(
            tickangle=-20, 
            tickfont=dict(size=12),
            title=""
        ),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fix_bar_label_clipping(fig_cust_count)
    st.plotly_chart(fig_cust_count, use_container_width=True)

st.markdown("")

# ROW 2: Average Revenue per Customer (Full-width Horizontal Bar)
st.markdown("##### 💰 Average Revenue per Customer by Membership Type")

avg_revenue_data = pd.DataFrame({
    'Membership Type': ['Single Sport', 'Multi Sport', 'Gold Membership'],
    'Avg Revenue': [avg_single, avg_multi, avg_gold],
    'Customer Count': [single_count, multi_count, gold_count]
})

fig_avg_rev = go.Figure()
fig_avg_rev.add_trace(go.Bar(
    y=avg_revenue_data['Membership Type'],
    x=avg_revenue_data['Avg Revenue'],
    orientation='h',
    text=avg_revenue_data['Avg Revenue'].map(lambda x: f"₹{x:,.0f}"),
    textposition='outside',
    marker=dict(
        color=['#fbbf24', '#3b82f6', '#10b981'],
        line=dict(width=0)
    ),
    hovertemplate='<b>%{y}</b><br>Avg Revenue: ₹%{x:,.0f}<extra></extra>'
))

fig_avg_rev.update_layout(
    height=350,
    margin=dict(l=180, r=120, t=30, b=50),
    xaxis_title="Average Package Value (₹)",
    xaxis=dict(tickfont=dict(size=12)),
    yaxis=dict(
        tickfont=dict(size=13),
        autorange="reversed"
    ),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fix_bar_label_clipping(fig_avg_rev)
st.plotly_chart(fig_avg_rev, use_container_width=True)

st.markdown("")

# ROW 3: KPI Cards for Customer Counts
st.markdown("##### 📈 Customer Distribution")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);">
        <div class="kpi-title">Single Sport Customers</div>
        <div class="kpi-value">{single_count}</div>
        <div class="kpi-subtitle">{(single_count/total_customers*100):.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);">
        <div class="kpi-title">Multi Sport Customers</div>
        <div class="kpi-value">{multi_count}</div>
        <div class="kpi-subtitle">{(multi_count/total_customers*100):.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
        <div class="kpi-title">Gold Members</div>
        <div class="kpi-value">{gold_count}</div>
        <div class="kpi-subtitle">{(gold_count/total_customers*100):.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ROW 4: KPI Cards for Total Revenue
st.markdown("##### 💵 Revenue Distribution")
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);">
        <div class="kpi-title">Single Sport Revenue</div>
        <div class="kpi-value">₹{single_rev:,.0f}</div>
        <div class="kpi-subtitle">MRR: ₹{single_mrr:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);">
        <div class="kpi-title">Multi Sport Revenue</div>
        <div class="kpi-value">₹{multi_rev:,.0f}</div>
        <div class="kpi-subtitle">MRR: ₹{multi_mrr:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
        <div class="kpi-title">Gold Membership Revenue</div>
        <div class="kpi-value">₹{gold_rev:,.0f}</div>
        <div class="kpi-subtitle">MRR: ₹{gold_mrr:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ----- FULL-WIDTH HORIZONTAL BAR: Revenue by Membership Type -----
st.markdown("##### 💰 Total Revenue Contribution")

rev_group = filtered_df.groupby('Membership Type')['product_net_revenue'].sum().reset_index()
rev_group.columns = ['Membership Type', 'Revenue']
rev_group = rev_group.sort_values('Revenue', ascending=True)

fig_rev_h = go.Figure()
fig_rev_h.add_trace(go.Bar(
    y=rev_group['Membership Type'],
    x=rev_group['Revenue'],
    orientation='h',
    text=rev_group['Revenue'].map(lambda x: f"₹{x:,.0f}"),
    textposition='outside',
    marker=dict(
        color=['#fbbf24', '#10b981', '#3b82f6'],
        line=dict(width=0)
    ),
    hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>'
))

fig_rev_h.update_layout(
    height=350,
    margin=dict(l=180, r=120, t=30, b=50),
    xaxis_title="Total Revenue (₹)",
    xaxis=dict(tickfont=dict(size=12)),
    yaxis=dict(tickfont=dict(size=13)),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fix_bar_label_clipping(fig_rev_h)
st.plotly_chart(fig_rev_h, use_container_width=True)

# ================== RENEWAL PIPELINE ==================
st.markdown("---")
st.markdown('<div class="metric-card">🔄 Renewal Pipeline & Follow-ups</div>', unsafe_allow_html=True)
st.markdown("")

col1b, col2b, col3b, col4b = st.columns(4)

with col1b:
    critical_df = filtered_df[filtered_df['Retention Risk'] == 'Critical']
    st.metric("🔴 Critical (Overdue)", len(critical_df), f"₹{critical_df['Due Amount'].sum():,.0f}")

with col2b:
    high_df = filtered_df[filtered_df['Retention Risk'] == 'High']
    st.metric("🟡 High Risk (≤14d)", len(high_df))

with col3b:
    active_df = filtered_df[filtered_df['Retention Risk'] == 'Active']
    st.metric("🟢 Active (>14d)", len(active_df))

with col4b:
    upcoming_30 = len(filtered_df[filtered_df['Days Until Renewal'].between(0, 30, inclusive="both")])
    st.metric("📅 Due in 30 Days", upcoming_30)

# Priority Follow-up Table
st.markdown("#### 📋 Priority Follow-up List")

action_df = filtered_df[filtered_df['Retention Risk'].isin(['Critical', 'High'])].copy()

if not action_df.empty:
    display_cols = [
        'wc_order_for_name', 'sport', 'school', 'place', 'academy_name',
        'product_net_revenue', 'Monthly Equivalent Fee', 'Months Overdue', 'Due Amount',
        'start_date', 'next_payment_date', 'Renewal Date', 'Days Until Renewal',
        'Retention Risk', 'phone', 'email'
    ]
    display_cols = [c for c in display_cols if c in action_df.columns]

    table_df = action_df[display_cols].sort_values(['Retention Risk', 'Days Until Renewal'])

    for date_col in ['start_date', 'next_payment_date', 'Renewal Date']:
        if date_col in table_df.columns:
            table_df[date_col] = pd.to_datetime(table_df[date_col]).dt.strftime('%d-%b-%Y')

    if 'product_net_revenue' in table_df.columns:
        table_df['product_net_revenue'] = table_df['product_net_revenue'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "")
    if 'Monthly Equivalent Fee' in table_df.columns:
        table_df['Monthly Equivalent Fee'] = table_df['Monthly Equivalent Fee'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
    if 'Due Amount' in table_df.columns:
        table_df['Due Amount'] = table_df['Due Amount'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "")

    def color_risk(val):
        if val == 'Critical':
            return 'background-color: #fecaca; font-weight: bold'
        elif val == 'High':
            return 'background-color: #fef3c7; font-weight: bold'
        return ''

    styled_df = table_df.style
    if 'Retention Risk' in table_df.columns:
        styled_df = styled_df.applymap(color_risk, subset=['Retention Risk'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    critical_count = len(action_df[action_df['Retention Risk'] == 'Critical'])
    high_count = len(action_df[action_df['Retention Risk'] == 'High'])
    multi_sport_risk = len(action_df[action_df['Is Multi-Sport']])

    st.warning(f"""
**📞 Action Summary**
- **{critical_count} Critical** customers: overdue and need immediate contact  
- **{high_count} High-risk** customers: due within the next 14 days  
- **{multi_sport_risk} Multi-sport customers** in this list (higher value retention)  
- Monthly revenue at risk: ₹{action_df['Monthly Equivalent Fee'].sum():,.2f}
    """)

    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Follow-up List",
        data=csv,
        file_name=f'followup_list_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
else:
    st.success("✅ No immediate follow-up required based on current filters.")

# ================== DETAILED ANALYTICS TABS ==================
st.markdown("---")
st.markdown('<div class="metric-card">📈 Detailed Analytics</div>', unsafe_allow_html=True)
st.markdown("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💰 Revenue Trends",
    "📍 Location Analysis",
    "🏫 School Analytics",
    "🏆 Academy Performance",
    "📱 Digital Analytics",
    "📦 Package Analysis"
])

# -------- TAB 1: REVENUE TRENDS --------
with tab1:
    st.subheader("📈 Monthly Revenue Trends")
    if not filtered_df.empty and 'start_date' in filtered_df.columns:
        trend_df = filtered_df.copy()
        trend_df['Month'] = trend_df['start_date'].dt.to_period('M').dt.to_timestamp()

        # Revenue & MRR
        monthly_rev = trend_df.groupby('Month').agg({
            'product_net_revenue': 'sum',
            'Monthly Equivalent Fee': 'sum'
        }).rename(columns={'product_net_revenue': 'Cash Revenue',
                           'Monthly Equivalent Fee': 'MRR'})

        # Orders per month
        monthly_orders = trend_df.groupby('Month').size().rename('Orders')

        # True NEW customers per month using student_id + first_order_date
        if 'student_id' in trend_df.columns and 'first_order_date' in trend_df.columns:
            first = trend_df[['student_id', 'first_order_date']].drop_duplicates().dropna(subset=['first_order_date'])
            first['Month'] = first['first_order_date'].dt.to_period('M').dt.to_timestamp()
            new_cust = first.groupby('Month').size().rename('New Customers')
        else:
            new_cust = pd.Series(dtype='float64', name='New Customers')

        monthly = pd.concat([monthly_rev, monthly_orders, new_cust], axis=1).fillna(0).reset_index()

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Bar(
            x=monthly['Month'],
            y=monthly['Cash Revenue'],
            name='Cash Revenue',
            marker_color='#3b82f6',
            yaxis='y1'
        ))

        fig_trend.add_trace(go.Scatter(
            x=monthly['Month'],
            y=monthly['MRR'],
            name='MRR',
            mode='lines+markers',
            yaxis='y1',
            line=dict(color='#10b981', width=3)
        ))

        # True new customers (red line)
        fig_trend.add_trace(go.Scatter(
            x=monthly['Month'],
            y=monthly['New Customers'],
            name='New Customers',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#ef4444', width=3)
        ))

        # Number of orders per month (purple dotted)
        fig_trend.add_trace(go.Scatter(
            x=monthly['Month'],
            y=monthly['Orders'],
            name='Orders',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#6b21a8', width=2, dash='dot')
        ))

        fig_trend.update_layout(
            xaxis=dict(title="Month"),
            yaxis=dict(title="Revenue (₹)", side='left'),
            yaxis2=dict(title="Customers / Orders", overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Revenue", f"₹{monthly['Cash Revenue'].sum():,.0f}")

        with c2:
            st.metric("Avg Monthly Revenue", f"₹{monthly['Cash Revenue'].mean():,.0f}")

        with c3:
            st.metric("Total MRR", f"₹{monthly['MRR'].sum():,.0f}")

        # Full monthly table + download
        st.markdown("##### 📊 Monthly Summary Table")
        st.dataframe(monthly, use_container_width=True)
        monthly_csv = monthly.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Monthly Summary CSV",
            data=monthly_csv,
            file_name=f'monthly_summary_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

# -------- TAB 2: LOCATION ANALYSIS --------
with tab2:
    st.subheader("📍 Location Analysis")

    if 'place' in filtered_df.columns:
        loc_df_raw = filtered_df[filtered_df['place'] != ''].copy()

        if not loc_df_raw.empty:
            # Aggregate by location, including number of academies in each place
            agg_dict = {
                'Total_Orders': ('wc_order_for_name', 'count'),
                'Total_Enrollments': ('product_qty', 'sum'),
                'Unique_Students': ('student_id', 'nunique'),
                'Total_Revenue': ('product_net_revenue', 'sum'),
                'MRR': ('Monthly Equivalent Fee', 'sum'),
                'Coupon_Discount': ('coupon_amount', 'sum'),
                'Outstanding_Dues': ('Due Amount', 'sum')
            }
            if 'academy_name' in loc_df_raw.columns:
                agg_dict['Academies_Count'] = ('academy_name', 'nunique')

            loc_agg = loc_df_raw.groupby('place').agg(**agg_dict).reset_index().rename(columns={'place': 'Location'})

            loc_agg = loc_agg.sort_values('Total_Revenue', ascending=False)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("##### Top Locations by Revenue")
                top_rev_loc = loc_agg.head(10)
                fig_loc_rev = px.bar(
                    top_rev_loc,
                    x='Location',
                    y='Total_Revenue',
                    text='Total_Revenue',
                    color='Total_Revenue',
                    color_continuous_scale='Blues'
                )
                fig_loc_rev.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig_loc_rev.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False,
                    height=400,
                    yaxis_title="Total Revenue (₹)"
                )
                fix_bar_label_clipping(fig_loc_rev)
                st.plotly_chart(fig_loc_rev, use_container_width=True)

            with c2:
                st.markdown("##### Top Locations by Enrollments")
                top_enrol_loc = loc_agg.sort_values('Total_Enrollments', ascending=False).head(10)
                fig_loc_enrol = px.bar(
                    top_enrol_loc,
                    x='Location',
                    y='Total_Enrollments',
                    text='Total_Enrollments'
                )
                fig_loc_enrol.update_traces(textposition='outside')
                fig_loc_enrol.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False,
                    height=400,
                    yaxis_title="Total Enrollments"
                )
                fix_bar_label_clipping(fig_loc_enrol)
                st.plotly_chart(fig_loc_enrol, use_container_width=True)

            # NEW: locations by number of academies
            if 'Academies_Count' in loc_agg.columns:
                st.markdown("---")
                st.markdown("##### Locations by Number of Academies")
                top_acad_loc = loc_agg.sort_values('Academies_Count', ascending=False).head(10)
                fig_loc_acad = px.bar(
                    top_acad_loc,
                    x='Location',
                    y='Academies_Count',
                    text='Academies_Count'
                )
                fig_loc_acad.update_traces(textposition='outside')
                fig_loc_acad.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False,
                    height=400,
                    yaxis_title="Number of Academies"
                )
                fix_bar_label_clipping(fig_loc_acad)
                st.plotly_chart(fig_loc_acad, use_container_width=True)

            st.markdown("---")
            st.markdown("##### 📊 Full Location Summary Table")
            loc_display = loc_agg.copy()
            for col in ['Total_Revenue', 'MRR', 'Coupon_Discount', 'Outstanding_Dues']:
                loc_display[col] = loc_display[col].map(lambda x: f"₹{x:,.0f}")
            st.dataframe(loc_display, use_container_width=True, hide_index=True)
            loc_csv = loc_agg.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Location Summary CSV",
                data=loc_csv,
                file_name=f'location_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )

        else:
            st.info("No location data available for the current filters.")
    else:
        st.info("Column `place` not found in data.")

# -------- TAB 3: SCHOOL ANALYTICS --------
with tab3:
    st.subheader("🏫 Detailed School Analytics")

    if 'school' in filtered_df.columns:
        school_df = filtered_df[filtered_df['school'] != ''].copy()

        if not school_df.empty:
            # Aggregate per school
            school_agg = school_df.groupby('school').agg(
                Unique_Students=('student_id', 'nunique'),
                Orders=('wc_order_for_name', 'count'),
                Total_Revenue=('product_net_revenue', 'sum'),
                MRR=('Monthly Equivalent Fee', 'sum'),
                Coupon_Discount=('coupon_amount', 'sum'),
                Outstanding_Dues=('Due Amount', 'sum')
            ).reset_index().rename(columns={'school': 'School'})

            # Membership mix per school
            membership_ct = school_df.pivot_table(
                index='school',
                columns='Membership Type',
                values='wc_order_for_name',
                aggfunc='count',
                fill_value=0
            ).reset_index().rename(columns={'school': 'School'})

            # Ensure columns exist
            for col in ['Single Sport', 'Multi Sport', 'Gold Membership']:
                if col not in membership_ct.columns:
                    membership_ct[col] = 0

            membership_ct = membership_ct.rename(columns={
                'Single Sport': 'Single Sport Count',
                'Multi Sport': 'Multi Sport Count',
                'Gold Membership': 'Gold Membership Count'
            })

            school_agg = school_agg.merge(membership_ct, on='School', how='left')

            # Avg revenue per student
            school_agg['Avg Revenue / Student'] = (
                school_agg['Total_Revenue'] / school_agg['Unique_Students'].replace(0, pd.NA)
            ).fillna(0)

            # Top 12 by revenue
            top_rev = school_agg.sort_values('Total_Revenue', ascending=False).head(12)

            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown("##### Top Schools by Total Revenue")
                fig_school_rev = px.bar(
                    top_rev,
                    x='School',
                    y='Total_Revenue',
                    text='Total_Revenue',
                    color='Total_Revenue',
                    color_continuous_scale='Blues'
                )
                fig_school_rev.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig_school_rev.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False,
                    height=420,
                    yaxis_title="Total Revenue (₹)"
                )
                fix_bar_label_clipping(fig_school_rev)
                st.plotly_chart(fig_school_rev, use_container_width=True)

            with c2:
                st.markdown("##### School Summary (Top 12 by Revenue)")
                display_cols = [
                    'School', 'Unique_Students', 'Orders',
                    'Total_Revenue', 'MRR',
                    'Coupon_Discount', 'Outstanding_Dues'
                ]
                table = top_rev[display_cols].copy()
                table['Total_Revenue'] = table['Total_Revenue'].map(lambda x: f"₹{x:,.0f}")
                table['MRR'] = table['MRR'].map(lambda x: f"₹{x:,.0f}")
                table['Coupon_Discount'] = table['Coupon_Discount'].map(lambda x: f"₹{x:,.0f}")
                table['Outstanding_Dues'] = table['Outstanding_Dues'].map(lambda x: f"₹{x:,.0f}")

                st.dataframe(table, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("##### Membership Mix by School (Top 8 by Orders)")

            top_orders = school_agg.sort_values('Orders', ascending=False).head(8)
            mix_df = top_orders.melt(
                id_vars='School',
                value_vars=['Single Sport Count', 'Multi Sport Count', 'Gold Membership Count'],
                var_name='Membership Type',
                value_name='Students'
            )
            mix_df['Membership Type'] = mix_df['Membership Type'].replace({
                'Single Sport Count': 'Single Sport',
                'Multi Sport Count': 'Multi Sport',
                'Gold Membership Count': 'Gold Membership'
            })

            fig_mix = px.bar(
                mix_df,
                x='School',
                y='Students',
                color='Membership Type',
                barmode='stack'
            )
            fig_mix.update_layout(
                xaxis_tickangle=-45,
                height=420,
                yaxis_title="Number of Students"
            )
            fix_bar_label_clipping(fig_mix)
            st.plotly_chart(fig_mix, use_container_width=True)

            st.markdown("---")
            s1, s2, s3, s4 = st.columns(4)

            top_school_by_rev = school_agg.sort_values('Total_Revenue', ascending=False).iloc[0]['School']
            top_school_by_students = school_agg.sort_values('Unique_Students', ascending=False).iloc[0]['School']
            max_avg_rev_row = school_agg.sort_values('Avg Revenue / Student', ascending=False).iloc[0]

            with s1:
                st.metric("Number of Active Schools", f"{school_agg['School'].nunique()}")

            with s2:
                st.metric("Top School by Revenue", top_school_by_rev)

            with s3:
                st.metric("Top School by Student Count", top_school_by_students)

            with s4:
                st.metric(
                    "Highest Avg Revenue / Student",
                    f"₹{max_avg_rev_row['Avg Revenue / Student']:,.0f}",
                    max_avg_rev_row['School']
                )

            # Full school table + download
            st.markdown("---")
            st.markdown("##### 📊 Full School Summary Table")
            school_disp = school_agg.copy()
            for col in ['Total_Revenue', 'MRR', 'Coupon_Discount', 'Outstanding_Dues', 'Avg Revenue / Student']:
                school_disp[col] = school_disp[col].map(lambda x: f"₹{x:,.0f}")
            st.dataframe(school_disp, use_container_width=True, hide_index=True)
            school_csv = school_agg.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download School Summary CSV",
                data=school_csv,
                file_name=f'school_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )

        else:
            st.info("No school-level data available for the current filters.")
    else:
        st.info("Column `school` not found in data.")

# -------- TAB 4: ACADEMY PERFORMANCE --------
with tab4:
    st.subheader("🏆 Academy Performance Analytics")

    if 'academy_name' in filtered_df.columns:
        acad_df = filtered_df[filtered_df['academy_name'] != ''].copy()

        if not acad_df.empty:
            # Aggregate per academy
            acad_agg = acad_df.groupby('academy_name').agg(
                Locations=('place', 'nunique'),
                Unique_Students=('student_id', 'nunique'),
                Orders=('wc_order_for_name', 'count'),
                Total_Enrollments=('product_qty', 'sum'),
                Total_Revenue=('product_net_revenue', 'sum'),
                MRR=('Monthly Equivalent Fee', 'sum'),
                Coupon_Discount=('coupon_amount', 'sum'),
                Outstanding_Dues=('Due Amount', 'sum'),
                Multi_Sport_Count=('Is Multi-Sport', 'sum'),
                Gold_Members=('Is Gold', 'sum')
            ).reset_index().rename(columns={'academy_name': 'Academy'})

            acad_agg['Avg Revenue / Student'] = (
                acad_agg['Total_Revenue'] / acad_agg['Unique_Students'].replace(0, pd.NA)
            ).fillna(0)

            # ===== KPI METRICS ROW =====
            st.markdown("##### 📊 Academy Overview Metrics")
            
            a1, a2, a3, a4 = st.columns(4)

            num_academies = acad_agg['Academy'].nunique()
            top_acad_by_rev = acad_agg.sort_values('Total_Revenue', ascending=False).iloc[0]
            top_acad_by_students = acad_agg.sort_values('Unique_Students', ascending=False).iloc[0]
            max_avg_rev_acad = acad_agg.sort_values('Avg Revenue / Student', ascending=False).iloc[0]

            with a1:
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                    <div class="kpi-title">Active Academies</div>
                    <div class="kpi-value">{num_academies}</div>
                    <div class="kpi-subtitle">Total academy partners</div>
                </div>
                """, unsafe_allow_html=True)

            with a2:
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                    <div class="kpi-title">Top by Revenue</div>
                    <div class="kpi-value" style="font-size: 1.3rem;">{top_acad_by_rev['Academy'][:20]}...</div>
                    <div class="kpi-subtitle">₹{top_acad_by_rev['Total_Revenue']:,.0f} revenue</div>
                </div>
                """, unsafe_allow_html=True)

            with a3:
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                    <div class="kpi-title">Top by Students</div>
                    <div class="kpi-value" style="font-size: 1.3rem;">{top_acad_by_students['Academy'][:20]}...</div>
                    <div class="kpi-subtitle">{top_acad_by_students['Unique_Students']} students</div>
                </div>
                """, unsafe_allow_html=True)

            with a4:
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                    <div class="kpi-title">Highest Avg Revenue/Student</div>
                    <div class="kpi-value">₹{max_avg_rev_acad['Avg Revenue / Student']:,.0f}</div>
                    <div class="kpi-subtitle">{max_avg_rev_acad['Academy'][:25]}...</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ===== TOP ACADEMIES BY REVENUE =====
            st.markdown("##### 💰 Top 15 Academies by Total Revenue")
            
            top_acad_rev_15 = acad_agg.sort_values('Total_Revenue', ascending=False).head(15)
            
            fig_acad_rev = go.Figure()
            fig_acad_rev.add_trace(go.Bar(
                y=top_acad_rev_15['Academy'],
                x=top_acad_rev_15['Total_Revenue'],
                orientation='h',
                text=top_acad_rev_15['Total_Revenue'].map(lambda x: f"₹{x:,.0f}"),
                textposition='outside',
                marker=dict(
                    color=top_acad_rev_15['Total_Revenue'],
                    colorscale='Blues',
                    showscale=False,
                    line=dict(width=0)
                ),
                hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>'
            ))
            fig_acad_rev.update_layout(
                height=600,
                margin=dict(l=280, r=120, t=30, b=50),
                xaxis_title="Total Revenue (₹)",
                yaxis_title="",
                xaxis=dict(tickfont=dict(size=11)),
                yaxis=dict(
                    autorange="reversed", 
                    tickfont=dict(size=11)
                ),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fix_bar_label_clipping(fig_acad_rev)
            st.plotly_chart(fig_acad_rev, use_container_width=True)

            st.markdown("---")

            # ===== TOP ACADEMIES BY STUDENTS =====
            st.markdown("##### 👥 Top 15 Academies by Unique Students")
            
            top_acad_students_15 = acad_agg.sort_values('Unique_Students', ascending=False).head(15)
            
            fig_acad_students = go.Figure()
            fig_acad_students.add_trace(go.Bar(
                y=top_acad_students_15['Academy'],
                x=top_acad_students_15['Unique_Students'],
                orientation='h',
                text=top_acad_students_15['Unique_Students'],
                textposition='outside',
                marker=dict(
                    color=top_acad_students_15['Unique_Students'],
                    colorscale='Greens',
                    showscale=False,
                    line=dict(width=0)
                ),
                hovertemplate='<b>%{y}</b><br>Students: %{x}<extra></extra>'
            ))
            fig_acad_students.update_layout(
                height=600,
                margin=dict(l=280, r=120, t=30, b=50),
                xaxis_title="Unique Students",
                yaxis_title="",
                xaxis=dict(tickfont=dict(size=11)),
                yaxis=dict(
                    autorange="reversed", 
                    tickfont=dict(size=11)
                ),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fix_bar_label_clipping(fig_acad_students)
            st.plotly_chart(fig_acad_students, use_container_width=True)

            st.markdown("---")

            # ===== MULTI-SPORT MEMBERS BY ACADEMY =====
            st.markdown("##### 🎯 Top 12 Academies by Multi-Sport Members")
            
            top_acad_multi = acad_agg.sort_values('Multi_Sport_Count', ascending=False).head(12).copy()
            top_acad_multi = top_acad_multi[top_acad_multi['Multi_Sport_Count'] > 0]

            if not top_acad_multi.empty:
                fig_acad_multi = go.Figure()
                fig_acad_multi.add_trace(go.Bar(
                    y=top_acad_multi['Academy'],
                    x=top_acad_multi['Multi_Sport_Count'],
                    orientation='h',
                    marker=dict(
                        color=top_acad_multi['Multi_Sport_Count'],
                        colorscale='Purples',
                        showscale=False,
                        line=dict(width=0)
                    ),
                    text=top_acad_multi['Multi_Sport_Count'],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Multi-Sport Members: %{x}<extra></extra>'
                ))
                fig_acad_multi.update_layout(
                    height=max(450, len(top_acad_multi) * 50),
                    margin=dict(l=280, r=120, t=30, b=50),
                    xaxis_title="Number of Multi-Sport Members",
                    yaxis_title="",
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(
                        autorange="reversed", 
                        tickfont=dict(size=11)
                    ),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fix_bar_label_clipping(fig_acad_multi)
                st.plotly_chart(fig_acad_multi, use_container_width=True)
            else:
                st.info("No academies with multi-sport members in current filter.")

            st.markdown("---")

            # ===== FULL ACADEMY TABLE + DOWNLOAD =====
            st.markdown("##### 📊 Complete Academy Performance Table")
            
            acad_disp = acad_agg.copy()
            for col in ['Total_Revenue', 'MRR', 'Coupon_Discount', 'Outstanding_Dues', 'Avg Revenue / Student']:
                acad_disp[col] = acad_disp[col].map(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(acad_disp, use_container_width=True, hide_index=True)
            
            acad_csv = acad_agg.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Academy Summary CSV",
                data=acad_csv,
                file_name=f'academy_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
            )

        else:
            st.info("No academy-level data available for the current filters.")
    else:
        st.info("Column `academy_name` not found in data.")

# -------- TAB 5: DIGITAL ANALYTICS --------
with tab5:
    st.subheader("📱 Digital Analytics")

    c1, c2 = st.columns(2)

    device_dist = pd.DataFrame()
    source_analysis = pd.DataFrame()

    with c1:
        if 'wc_order_attribution_device_type' in filtered_df.columns:
            st.markdown("##### Device Type Distribution")
            device_dist = filtered_df[filtered_df['wc_order_attribution_device_type'] != '']['wc_order_attribution_device_type'].value_counts().reset_index()
            device_dist.columns = ['Device', 'Count']

            if not device_dist.empty:
                fig_device = px.pie(
                    device_dist,
                    values='Count',
                    names='Device',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_device.update_traces(textposition='inside', textinfo='percent+label+value')
                fig_device.update_layout(height=350)
                st.plotly_chart(fig_device, use_container_width=True)
            else:
                st.info("No device data for current filters.")

    with c2:
        if 'wc_order_attribution_utm_source' in filtered_df.columns:
            st.markdown("##### Traffic Source Performance")
            source_analysis = filtered_df[filtered_df['wc_order_attribution_utm_source'] != ''].groupby('wc_order_attribution_utm_source').agg({
                'wc_order_for_name': 'count',
                'product_net_revenue': 'sum'
            }).reset_index()
            source_analysis.columns = ['Source', 'Customers', 'Revenue']

            if not source_analysis.empty:
                fig_source = px.bar(
                    source_analysis,
                    x='Source',
                    y='Revenue',
                    text='Revenue',
                    color='Source',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_source.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig_source.update_layout(showlegend=False, height=350)
                fix_bar_label_clipping(fig_source)
                st.plotly_chart(fig_source, use_container_width=True)
            else:
                st.info("No source data for current filters.")

    # Full tables + downloads
    st.markdown("---")
    col_da1, col_da2 = st.columns(2)

    with col_da1:
        st.markdown("##### 📊 Device Summary Table")
        if not device_dist.empty:
            st.dataframe(device_dist, use_container_width=True, hide_index=True)
            dev_csv = device_dist.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Device Summary CSV",
                data=dev_csv,
                file_name=f'device_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.info("No device summary to show.")

    with col_da2:
        st.markdown("##### 📊 Traffic Source Summary Table")
        if not source_analysis.empty:
            src_display = source_analysis.copy()
            src_display['Revenue'] = src_display['Revenue'].map(lambda x: f"₹{x:,.0f}")
            st.dataframe(src_display, use_container_width=True, hide_index=True)
            src_csv = source_analysis.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Source Summary CSV",
                data=src_csv,
                file_name=f'source_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.info("No traffic source summary to show.")

# -------- TAB 6: PACKAGE ANALYSIS --------
with tab6:
    st.subheader("📦 Package Duration Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Package Distribution")
        pkg_dist = filtered_df['Package Duration (Months)'].value_counts().reset_index()
        pkg_dist.columns = ['Duration', 'Count']

        pkg_map = {1: '1 Month', 2: '2 Months', 3: '3 Months', 4: '4 Months',
                   5: '5 Months', 6: '6 Months', 12: 'Yearly'}
        pkg_dist['Duration Label'] = pkg_dist['Duration'].map(lambda x: pkg_map.get(x, f'{x} Months'))

        fig_pkg = px.pie(
            pkg_dist,
            values='Count',
            names='Duration Label',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pkg.update_traces(textposition='inside', textinfo='percent+label')
        fig_pkg.update_layout(height=400)
        st.plotly_chart(fig_pkg, use_container_width=True)

    with c2:
        st.markdown("##### Revenue by Package Duration")
        pkg_rev = filtered_df.groupby('Package Duration (Months)').agg({
            'product_net_revenue': 'sum',
            'wc_order_for_name': 'count'
        }).reset_index()
        pkg_rev.columns = ['Duration', 'Revenue', 'Count']
        pkg_rev['Duration Label'] = pkg_rev['Duration'].map(lambda x: pkg_map.get(x, f'{x}M'))

        fig_pkg_rev = px.bar(
            pkg_rev,
            x='Duration Label',
            y='Revenue',
            text='Revenue',
            color='Revenue',
            color_continuous_scale='Greens'
        )
        fig_pkg_rev.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig_pkg_rev.update_layout(showlegend=False, height=400)
        fix_bar_label_clipping(fig_pkg_rev)
        st.plotly_chart(fig_pkg_rev, use_container_width=True)

    # Full package summary table + download
    st.markdown("---")
    st.markdown("##### 📊 Package Summary Table")
    pkg_summary = pkg_rev.copy()
    pkg_summary['Revenue'] = pkg_summary['Revenue'].map(lambda x: f"₹{x:,.0f}")
    st.dataframe(pkg_summary, use_container_width=True, hide_index=True)
    pkg_csv = pkg_rev.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Package Summary CSV",
        data=pkg_csv,
        file_name=f'package_summary_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

# ================== EXPORT ==================
st.markdown("---")
st.subheader("📥 Export Reports")

e1, e2, e3 = st.columns(3)

with e1:
    full_csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Full Report (Filtered)",
        data=full_csv,
        file_name=f'sports_academy_full_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

with e2:
    critical_csv = filtered_df[filtered_df['Retention Risk'] == 'Critical'].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Critical Renewals List",
        data=critical_csv,
        file_name=f'critical_renewals_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

with e3:
    multi_sport_csv = filtered_df[filtered_df['Is Multi-Sport']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Multi-Sport Customers",
        data=multi_sport_csv,
        file_name=f'multi_sport_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

# ================== FOOTER ==================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>Sports Academy Management System v2.0</h4>
    <p>Dashboard Updated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
    <p>Total Records: {len(df):,} | Filtered Records: {len(filtered_df):,}</p>
    <p>🎯 Focus Areas: Multi-sport growth • Renewal management • Location expansion</p>
    <p>💾 Connected to MySQL Database: {DB_NAME}</p>
</div>
""", unsafe_allow_html=True)
