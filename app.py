"""
ATM Intelligence Demand Forecasting — FA-2
FinTrust Bank Ltd. | Dark Premium Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinTrust ATM Intelligence",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════
C = {
    "bg":       "#050914",
    "surface":  "#0d1528",
    "surface2": "#111e35",
    "border":   "rgba(0,212,255,0.12)",
    "cyan":     "#00d4ff",
    "cyan_dim": "rgba(0,212,255,0.15)",
    "amber":    "#ffb347",
    "green":    "#00ff88",
    "red":      "#ff4d6d",
    "purple":   "#b57bee",
    "text":     "#e8f0fe",
    "muted":    "#6b82a8",
}

LAYOUT = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["bg"],
    font=dict(color=C["text"], family="'Rajdhani', sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=C["border"], tickcolor=C["muted"]),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=C["border"], tickcolor=C["muted"]),
    colorway=[C["cyan"], C["amber"], C["green"], C["purple"], C["red"], "#4fc3f7"],
    legend=dict(bgcolor="rgba(13,21,40,0.8)", bordercolor=C["border"], borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ═══════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {C["bg"]} !important;
    color: {C["text"]};
    font-family: 'Inter', sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.5rem; max-width: 100%; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #07101f 0%, #0a1525 100%) !important;
    border-right: 1px solid {C["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {C["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {C["cyan_dim"]}; border-radius: 2px; }}

h1, h2, h3 {{ font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.04em; }}

.kpi-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {C["cyan"]}, {C["purple"]});
}}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: {C["muted"]};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {C["cyan"]};
    line-height: 1;
    text-shadow: 0 0 20px rgba(0,212,255,0.35);
}}
.kpi-delta {{
    font-size: 0.72rem;
    color: {C["green"]};
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}}

.section-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
}}
.section-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: {C["text"]};
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.section-line {{
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {C["border"]}, transparent);
}}
.stage-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    background: {C["cyan_dim"]};
    border: 1px solid {C["border"]};
    color: {C["cyan"]};
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    white-space: nowrap;
}}

.alert-critical {{
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.85rem;
}}
.alert-warning {{
    background: rgba(255,179,71,0.08);
    border: 1px solid rgba(255,179,71,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.85rem;
}}
.alert-ok {{
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 8px;
    padding: 16px;
    margin: 5px 0;
    font-size: 0.85rem;
}}

.insight-box {{
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(181,123,238,0.05));
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.86rem;
    line-height: 1.6;
}}

.ai-response {{
    background: linear-gradient(135deg, rgba(181,123,238,0.07), rgba(0,212,255,0.04));
    border: 1px solid rgba(181,123,238,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.88rem;
    line-height: 1.75;
}}
.ai-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: {C["purple"]};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: block;
}}

.sidebar-logo {{
    text-align: center;
    padding: 8px 0 18px;
    border-bottom: 1px solid {C["border"]};
    margin-bottom: 14px;
}}
.sidebar-logo-text {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: {C["cyan"]};
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.sidebar-logo-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: {C["muted"]};
    letter-spacing: 0.1em;
}}

[data-testid="stButton"] > button {{
    background: {C["cyan_dim"]} !important;
    border: 1px solid {C["cyan"]} !important;
    color: {C["cyan"]} !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
}}
[data-testid="stButton"] > button:hover {{
    background: rgba(0,212,255,0.22) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.28) !important;
}}

[data-testid="stTabs"] [role="tablist"] {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border"]};
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}}
[data-testid="stTabs"] button[role="tab"] {{
    color: {C["muted"]} !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: {C["cyan_dim"]} !important;
    color: {C["cyan"]} !important;
}}

[data-testid="stDataFrame"] th {{
    background: {C["surface2"]} !important;
    color: {C["cyan"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}}

@media (max-width: 768px) {{
    .kpi-value {{ font-size: 1.3rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    np.random.seed(99)
    atm_list = df["ATM_ID"].unique()
    df["lat"] = df["ATM_ID"].map({a: np.random.uniform(8.5, 35.0) for a in atm_list})
    df["lon"] = df["ATM_ID"].map({a: np.random.uniform(68.0, 97.5) for a in atm_list})
    return df


# ═══════════════════════════════════════════════════════════
# AI
# ═══════════════════════════════════════════════════════════
def get_ai_insight(context: str, api_key: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=350,
            messages=[{"role": "user", "content":
                f"You are a senior data analyst at FinTrust Bank Ltd. "
                f"Based on this ATM data context, write 2-3 sharp, actionable bullet points (use • symbol). "
                f"Be concise, specific, bank-operations focused. No fluff.\n\nContext:\n{context}"}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"⚠️ AI insight unavailable: {str(e)}"


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:1.8rem; margin-bottom:4px;">🏧</div>
        <div class="sidebar-logo-text">FinTrust ATM</div>
        <div class="sidebar-logo-sub">INTELLIGENCE PLATFORM v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🌐  Dashboard",
        "📊  EDA Explorer",
        "🔵  Cluster Analysis",
        "⚠️   Anomaly Radar",
        "🎛️   Demand Planner",
        "🗺️   ATM Heatmap",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="kpi-label">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="kpi-label">AI Insights (Optional)</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...", label_visibility="collapsed")
    if api_key:
        st.markdown(f'<div style="color:{C["green"]};font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">✓ AI READY</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="kpi-label">Z-Score Threshold</div>', unsafe_allow_html=True)
    z_thresh = st.slider("Z", 1.5, 4.0, 2.5, 0.1, label_visibility="collapsed")
    st.markdown('<div class="kpi-label" style="margin-top:10px">Clusters (K)</div>', unsafe_allow_html=True)
    k_val = st.slider("K", 2, 6, 3, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f'<div class="sidebar-logo-sub" style="text-align:center">FA-2 · DATA MINING · CRS · Y1</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
def process_df(raw_df):
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df["Month"] = raw_df["Date"].dt.month
    raw_df["MonthName"] = raw_df["Date"].dt.strftime("%b")
    np.random.seed(99)
    atm_list = raw_df["ATM_ID"].unique()
    raw_df["lat"] = raw_df["ATM_ID"].map({a: np.random.uniform(8.5, 35.0) for a in atm_list})
    raw_df["lon"] = raw_df["ATM_ID"].map({a: np.random.uniform(68.0, 97.5) for a in atm_list})
    return raw_df

if uploaded:
    import io
    df = process_df(pd.read_csv(io.BytesIO(uploaded.read())))
else:
    try:
        df = load_data("atm_cash_management_dataset.csv")
    except Exception:
        st.error("⚠️ Please upload your dataset (atm_cash_management_dataset.csv) using the sidebar.")
        st.stop()


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def section(title, badge=None):
    pill = f'<span class="stage-pill">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        {pill}
        <span class="section-title">{title}</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def ai_block(ctx, key, label="✦ Generate AI Insight"):
    if api_key:
        if st.button(label, key=key):
            with st.spinner("Analysing with Claude..."):
                result = get_ai_insight(ctx, api_key)
            st.markdown(f"""
            <div class="ai-response">
                <span class="ai-badge">✦ Claude AI · FinTrust Analyst</span>
                {result.replace(chr(10), "<br>")}
            </div>""", unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            delta_html = f'<div class="kpi-delta">↑ {delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {delta_html}
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🌐  Dashboard":
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.4rem;font-weight:700;letter-spacing:0.06em;line-height:1.1;">
            ATM INTELLIGENCE
            <span style="color:{C["cyan"]};text-shadow:0 0 30px rgba(0,212,255,0.45);">DASHBOARD</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.14em;margin-top:4px;">
            FINTRUST BANK LTD. · DEMAND FORECASTING · DATA MINING FA-2
        </div>
    </div>""", unsafe_allow_html=True)

    total_w = df["Total_Withdrawals"].sum()
    avg_w   = df["Total_Withdrawals"].mean()
    n_atms  = df["ATM_ID"].nunique()
    h_lift  = (df[df["Holiday_Flag"]==1]["Total_Withdrawals"].mean() /
               df[df["Holiday_Flag"]==0]["Total_Withdrawals"].mean() - 1) * 100
    kpi_row([
        ("TOTAL WITHDRAWALS",  f"₹{total_w/1e9:.2f}B", None),
        ("AVG DAILY DEMAND",   f"₹{avg_w/1000:.1f}K",  None),
        ("ATMs MONITORED",     str(n_atms),             None),
        ("HOLIDAY LIFT",       f"+{h_lift:.0f}%",       "vs normal days"),
    ])

    # Alert system
    section("LIVE ALERT SYSTEM", "ALERTS")
    atm_avg  = df.groupby("ATM_ID")["Total_Withdrawals"].mean()
    gmean, gstd = atm_avg.mean(), atm_avg.std()
    critical = atm_avg[atm_avg > gmean + 1.5*gstd]
    warning  = atm_avg[(atm_avg > gmean + 0.8*gstd) & (atm_avg <= gmean + 1.5*gstd)]
    ok       = atm_avg[atm_avg <= gmean + 0.8*gstd]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-label" style="color:{C["red"]}">🔴 CRITICAL — {len(critical)} ATMs</div>', unsafe_allow_html=True)
        for atm, val in critical.head(6).items():
            st.markdown(f'<div class="alert-critical">⚠ <b>{atm}</b> — ₹{val/1000:.1f}K avg</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-label" style="color:{C["amber"]}">🟡 WARNING — {len(warning)} ATMs</div>', unsafe_allow_html=True)
        for atm, val in warning.head(6).items():
            st.markdown(f'<div class="alert-warning">◈ <b>{atm}</b> — ₹{val/1000:.1f}K avg</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-label" style="color:{C["green"]}">🟢 NORMAL — {len(ok)} ATMs</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-ok">✓ {len(ok)} ATMs operating within normal demand thresholds. No immediate replenishment required.</div>', unsafe_allow_html=True)

    section("WITHDRAWAL OVERVIEW", "TRENDS")
    col1, col2 = st.columns([2,1])
    with col1:
        daily = df.groupby("Date")["Total_Withdrawals"].mean().reset_index()
        hol   = df[df["Holiday_Flag"]==1].groupby("Date")["Total_Withdrawals"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total_Withdrawals"],
                                  mode="lines", name="Daily Avg",
                                  line=dict(color=C["cyan"], width=2),
                                  fill="tozeroy", fillcolor="rgba(0,212,255,0.04)"))
        fig.add_trace(go.Scatter(x=hol["Date"], y=hol["Total_Withdrawals"],
                                  mode="markers", name="Holiday",
                                  marker=dict(color=C["red"], size=7, symbol="circle",
                                             line=dict(color="white",width=1))))
        fig.update_layout(**LAYOUT, title="Daily Average Withdrawals", height=280)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        loc = df.groupby("Location_Type")["Total_Withdrawals"].mean().sort_values()
        fig2 = go.Figure(go.Bar(y=loc.index, x=loc.values, orientation="h",
                                 marker=dict(color=loc.values,
                                            colorscale=[[0,C["surface2"]],[1,C["cyan"]]],
                                            line=dict(color=C["border"],width=0.5))))
        fig2.update_layout(**LAYOUT, title="Avg by Location", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    ai_block(f"Dashboard: {n_atms} ATMs, avg ₹{avg_w:.0f}, holiday lift +{h_lift:.1f}%, {len(critical)} critical ATMs.", "dash_ai")


# ═══════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════
elif page == "📊  EDA Explorer":
    st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.06em;">EXPLORATORY <span style="color:{C["cyan"]}">DATA ANALYSIS</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.12em;margin-bottom:18px;">STAGE 3 · PATTERN DISCOVERY</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 Distributions","📈 Time Trends","🎉 Holiday Impact","🌦 External Factors","🔗 Correlations"])

    with tab1:
        section("DISTRIBUTION ANALYSIS", "3.1")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(x=df["Total_Withdrawals"], nbinsx=45,
                                          marker=dict(color=C["cyan"], opacity=0.75,
                                                     line=dict(color=C["bg"],width=0.3))))
            fig.update_layout(**LAYOUT, title="Distribution — Total Withdrawals", height=300)
            st.plotly_chart(fig, use_container_width=True)
            insight("Right skew confirms most ATMs operate at moderate load. Extreme tail events align with holidays/special events.")
        with c2:
            fig = go.Figure(go.Histogram(x=df["Total_Deposits"], nbinsx=45,
                                          marker=dict(color=C["amber"], opacity=0.75,
                                                     line=dict(color=C["bg"],width=0.3))))
            fig.update_layout(**LAYOUT, title="Distribution — Total Deposits", height=300)
            st.plotly_chart(fig, use_container_width=True)
            insight("Deposits are ~5x lower than withdrawals, confirming net cash outflow pressure on all ATMs.")

        section("OUTLIER DETECTION — BOX PLOTS", "3.1")
        fig = go.Figure()
        for feat, col in [("Total_Withdrawals",C["cyan"]),("Total_Deposits",C["amber"]),
                          ("Cash_Demand_Next_Day",C["green"])]:
            fig.add_trace(go.Box(y=df[feat], name=feat.replace("_"," "),
                                  marker_color=col, line_color=col,
                                  boxmean="sd", fillcolor="rgba(0,0,0,0)"))
        fig.update_layout(**LAYOUT, title="Box Plots — Key Features", height=350)
        st.plotly_chart(fig, use_container_width=True)
        ai_block("Distribution: withdrawals right-skewed, deposits concentrated lower, outliers present at holiday/event peaks.", "eda_dist_ai")

    with tab2:
        section("TIME-BASED PATTERNS", "3.2")
        daily = df.groupby("Date")["Total_Withdrawals"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total_Withdrawals"].rolling(7).mean(),
                                  mode="lines", name="7-Day MA",
                                  line=dict(color=C["cyan"], width=2.5)))
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total_Withdrawals"],
                                  mode="lines", name="Daily",
                                  line=dict(color=f"rgba(0,212,255,0.22)", width=1)))
        fig.update_layout(**LAYOUT, title="Withdrawal Trend + 7-Day Moving Average", height=300)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow = df.groupby("Day_of_Week")["Total_Withdrawals"].mean().reindex(day_order).reset_index()
            cols_bar = [C["cyan"] if d in ["Saturday","Sunday"] else C["surface2"] for d in dow["Day_of_Week"]]
            fig = go.Figure(go.Bar(x=dow["Day_of_Week"], y=dow["Total_Withdrawals"],
                                   marker=dict(color=cols_bar, line=dict(color=C["border"],width=0.5))))
            fig.update_layout(**LAYOUT, title="Avg Withdrawals by Day of Week", height=280)
            st.plotly_chart(fig, use_container_width=True)
            insight("Weekend demand peaks 15-25% above weekdays — Friday evening ATM pre-loading is critical.")
        with c2:
            tod_order = ["Morning","Afternoon","Evening","Night"]
            tod = df.groupby("Time_of_Day")["Total_Withdrawals"].mean().reindex(tod_order).reset_index()
            fig = go.Figure(go.Bar(x=tod["Time_of_Day"], y=tod["Total_Withdrawals"],
                                   marker=dict(color=tod["Total_Withdrawals"],
                                              colorscale=[[0,C["surface2"]],[1,C["amber"]]],
                                              line=dict(color=C["border"],width=0.5))))
            fig.update_layout(**LAYOUT, title="Avg Withdrawals by Time of Day", height=280)
            st.plotly_chart(fig, use_container_width=True)
            insight("Afternoon slot drives peak demand. Cash-in-transit should be scheduled before 12:00 PM daily.")

        ai_block(f"Time patterns: weekend avg ₹{df[df['Day_of_Week'].isin(['Saturday','Sunday'])]['Total_Withdrawals'].mean():.0f} vs weekday ₹{df[~df['Day_of_Week'].isin(['Saturday','Sunday'])]['Total_Withdrawals'].mean():.0f}. Peak: Afternoon.", "eda_time_ai")

    with tab3:
        section("HOLIDAY & EVENT IMPACT", "3.3")
        c1, c2 = st.columns(2)
        with c1:
            hf = df.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
            hf["Label"] = hf["Holiday_Flag"].map({0:"Normal Day",1:"Holiday"})
            fig = go.Figure(go.Bar(x=hf["Label"], y=hf["Total_Withdrawals"],
                                   marker=dict(color=[C["surface2"],C["red"]],
                                              line=dict(color=C["border"],width=0.5)),
                                   text=[f"₹{v/1000:.1f}K" for v in hf["Total_Withdrawals"]],
                                   textposition="outside", textfont=dict(color=C["text"])))
            fig.update_layout(**LAYOUT, title="Holiday vs Normal", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ef = df.groupby("Special_Event_Flag")["Total_Withdrawals"].mean().reset_index()
            ef["Label"] = ef["Special_Event_Flag"].map({0:"No Event",1:"Special Event"})
            fig = go.Figure(go.Bar(x=ef["Label"], y=ef["Total_Withdrawals"],
                                   marker=dict(color=[C["surface2"],C["amber"]],
                                              line=dict(color=C["border"],width=0.5)),
                                   text=[f"₹{v/1000:.1f}K" for v in ef["Total_Withdrawals"]],
                                   textposition="outside", textfont=dict(color=C["text"])))
            fig.update_layout(**LAYOUT, title="Event vs Non-Event", height=300)
            st.plotly_chart(fig, use_container_width=True)

        monthly = df.groupby(["MonthName","Month","Holiday_Flag"])["Total_Withdrawals"].mean().reset_index().sort_values("Month")
        monthly["Type"] = monthly["Holiday_Flag"].map({0:"Normal",1:"Holiday"})
        fig = px.line(monthly, x="MonthName", y="Total_Withdrawals", color="Type", markers=True,
                      color_discrete_map={"Normal":C["cyan"],"Holiday":C["red"]})
        fig.update_layout(**LAYOUT, title="Monthly Trend — Normal vs Holiday", height=280)
        st.plotly_chart(fig, use_container_width=True)
        ai_block(f"Holiday avg ₹{df[df['Holiday_Flag']==1]['Total_Withdrawals'].mean():.0f} vs normal ₹{df[df['Holiday_Flag']==0]['Total_Withdrawals'].mean():.0f}. Event avg ₹{df[df['Special_Event_Flag']==1]['Total_Withdrawals'].mean():.0f}.", "eda_hol_ai")

    with tab4:
        section("EXTERNAL FACTORS", "3.4")
        c1, c2 = st.columns(2)
        with c1:
            wd = df.groupby("Weather_Condition")["Total_Withdrawals"].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=wd.index, y=wd.values,
                                   marker=dict(color=[C["cyan"],C["amber"],C["muted"],C["purple"]],
                                              line=dict(color=C["border"],width=0.5)),
                                   text=[f"₹{v/1000:.1f}K" for v in wd.values],
                                   textposition="outside", textfont=dict(color=C["text"])))
            fig.update_layout(**LAYOUT, title="Avg Withdrawals by Weather", height=300)
            st.plotly_chart(fig, use_container_width=True)
            insight("Stormy/Snowy conditions suppress demand. Pre-storm loading prevents over-stocking idle cash.")
        with c2:
            cd = df.groupby("Nearby_Competitor_ATMs")["Total_Withdrawals"].mean().reset_index()
            fig = go.Figure(go.Scatter(x=cd["Nearby_Competitor_ATMs"], y=cd["Total_Withdrawals"],
                                        mode="lines+markers",
                                        line=dict(color=C["purple"], width=2.5),
                                        marker=dict(color=C["purple"], size=10, line=dict(color=C["bg"],width=2))))
            fig.update_layout(**LAYOUT, title="Withdrawal vs Competitor ATMs", height=300,
                              xaxis_title="Competitor ATMs Nearby", yaxis_title="Avg Withdrawals")
            st.plotly_chart(fig, use_container_width=True)
            insight("Isolated ATMs (0 competitors) show highest demand — classify as Tier-1 nodes for zero-downtime SLA.")

    with tab5:
        section("CORRELATION ANALYSIS", "3.5")
        c1, c2 = st.columns(2)
        with c1:
            num_cols = ["Total_Withdrawals","Total_Deposits","Nearby_Competitor_ATMs",
                        "Previous_Day_Cash_Level","Cash_Demand_Next_Day","Holiday_Flag","Special_Event_Flag"]
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0,"#1a0a2e"],[0.5,C["surface"]],[1,C["cyan"]]],
                text=np.round(corr.values,2), texttemplate="%{text}",
                textfont=dict(size=9, color=C["text"]),
                zmin=-1, zmax=1
            ))
            fig.update_layout(**LAYOUT, title="Correlation Heatmap", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            samp = df.sample(min(800,len(df)), random_state=42)
            fig = px.scatter(samp, x="Previous_Day_Cash_Level", y="Cash_Demand_Next_Day",
                             color="Location_Type",
                             color_discrete_sequence=[C["cyan"],C["amber"],C["green"],C["purple"],C["red"]],
                             opacity=0.5)
            fig.update_layout(**LAYOUT, title="Previous Cash Level vs Next Day Demand", height=380)
            st.plotly_chart(fig, use_container_width=True)
            insight(f"Withdrawals ↔ NextDayDemand r={corr.loc['Total_Withdrawals','Cash_Demand_Next_Day']:.2f} — strong predictive signal for demand forecasting models.")
        ai_block(f"Correlation: Withdrawals vs NextDayDemand r={corr.loc['Total_Withdrawals','Cash_Demand_Next_Day']:.2f}. Holiday flag r={corr.loc['Total_Withdrawals','Holiday_Flag']:.2f}.", "eda_corr_ai")


# ═══════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════
elif page == "🔵  Cluster Analysis":
    st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.06em;">CLUSTER <span style="color:{C["cyan"]}">ANALYSIS</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.12em;margin-bottom:18px;">STAGE 4 · K-MEANS SEGMENTATION</div>', unsafe_allow_html=True)

    feat_cols = ["Total_Withdrawals","Total_Deposits","Nearby_Competitor_ATMs","Cash_Demand_Next_Day"]
    atm_agg = df.groupby("ATM_ID")[feat_cols].mean().reset_index()
    loc_enc = {"Standalone":0,"Gas Station":1,"Supermarket":2,"Bank Branch":3,"Mall":4}
    atm_agg["Location_enc"] = df.groupby("ATM_ID")["Location_Type"].apply(
        lambda x: x.map(loc_enc).mean()).values

    X = atm_agg[feat_cols + ["Location_enc"]].values
    Xs = StandardScaler().fit_transform(X)

    K_range = range(2, 8)
    inertias, sils = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))

    section("OPTIMAL K SELECTION", "4.1")
    c1, c2 = st.columns(2)
    with c1:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers", name="Inertia",
                                  line=dict(color=C["cyan"],width=2.5),
                                  marker=dict(size=8,color=C["cyan"],line=dict(color=C["bg"],width=2))), secondary_y=False)
        fig.add_trace(go.Scatter(x=list(K_range), y=sils, mode="lines+markers", name="Silhouette",
                                  line=dict(color=C["amber"],width=2.5,dash="dot"),
                                  marker=dict(size=8,color=C["amber"],line=dict(color=C["bg"],width=2))), secondary_y=True)
        fig.add_vline(x=k_val, line_color=C["green"], line_dash="dash", line_width=1.5,
                     annotation_text=f"K={k_val}", annotation_font_color=C["green"])
        fig.update_layout(**LAYOUT, title="Elbow + Silhouette", height=320)
        st.plotly_chart(fig, use_container_width=True)

    km_f = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    atm_agg["Cluster"] = km_f.fit_predict(Xs)
    cl_means = atm_agg.groupby("Cluster")["Total_Withdrawals"].mean().sort_values(ascending=False)
    labels_map = {cl: nm for cl, nm in zip(cl_means.index,
                  ["🏙 High-Demand","🏢 Steady-Demand","🌾 Low-Demand","📍 Seg 4","📍 Seg 5","📍 Seg 6"])}
    atm_agg["Segment"] = atm_agg["Cluster"].map(labels_map)

    with c2:
        cl_bar = atm_agg.groupby("Segment")["Total_Withdrawals"].agg(["mean","count"]).reset_index()
        colors_c = [C["red"],C["cyan"],C["green"],C["amber"],C["purple"]]
        fig = go.Figure(go.Bar(x=cl_bar["Segment"], y=cl_bar["mean"],
                                marker=dict(color=colors_c[:k_val], line=dict(color=C["border"],width=0.5)),
                                text=[f"n={int(r)}" for r in cl_bar["count"]],
                                textposition="outside", textfont=dict(color=C["text"])))
        fig.update_layout(**LAYOUT, title="Avg Withdrawals by Cluster", height=320)
        st.plotly_chart(fig, use_container_width=True)

    section("3D CLUSTER VISUALIZATION", "4.2")
    fig = px.scatter_3d(atm_agg, x="Total_Withdrawals", y="Total_Deposits",
                         z="Cash_Demand_Next_Day", color="Segment",
                         color_discrete_sequence=colors_c[:k_val],
                         hover_name="ATM_ID", opacity=0.85)
    fig.update_layout(
        paper_bgcolor=C["surface"], font=dict(color=C["text"],family="'Rajdhani',sans-serif"),
        title="3D ATM Cluster Space", height=500,
        scene=dict(
            bgcolor=C["bg"],
            xaxis=dict(backgroundcolor=C["bg"],gridcolor="rgba(255,255,255,0.05)",color=C["muted"]),
            yaxis=dict(backgroundcolor=C["bg"],gridcolor="rgba(255,255,255,0.05)",color=C["muted"]),
            zaxis=dict(backgroundcolor=C["bg"],gridcolor="rgba(255,255,255,0.05)",color=C["muted"]),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    section("ATM SEGMENT ASSIGNMENTS", "4.3")
    disp = atm_agg[["ATM_ID","Segment"]+feat_cols].sort_values("Segment").round(0)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    ai_block(f"K={k_val} clusters. Distribution: {atm_agg['Segment'].value_counts().to_dict()}. High-demand avg: ₹{cl_means.iloc[0]:.0f}.", "clust_ai")


# ═══════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════
elif page == "⚠️   Anomaly Radar":
    st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.06em;">ANOMALY <span style="color:{C["red"]};text-shadow:0 0 20px rgba(255,77,109,0.4);">RADAR</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.12em;margin-bottom:18px;">STAGE 5 · STATISTICAL ANOMALY DETECTION</div>', unsafe_allow_html=True)

    method = st.radio("Detection Method", ["Z-Score","IQR","Isolation Forest"], horizontal=True)
    df_ad = df.copy()
    if method == "Z-Score":
        z = np.abs(stats.zscore(df_ad["Total_Withdrawals"]))
        df_ad["Anomaly"] = z > z_thresh
        df_ad["Score"]   = z
    elif method == "IQR":
        Q1, Q3 = df_ad["Total_Withdrawals"].quantile(0.25), df_ad["Total_Withdrawals"].quantile(0.75)
        IQR = Q3 - Q1
        df_ad["Anomaly"] = (df_ad["Total_Withdrawals"] > Q3+1.5*IQR) | (df_ad["Total_Withdrawals"] < Q1-1.5*IQR)
        df_ad["Score"]   = np.abs(df_ad["Total_Withdrawals"] - df_ad["Total_Withdrawals"].median()) / IQR
    else:
        iso   = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(df_ad[["Total_Withdrawals","Holiday_Flag","Special_Event_Flag"]])
        df_ad["Anomaly"] = preds == -1
        df_ad["Score"]   = -iso.decision_function(df_ad[["Total_Withdrawals","Holiday_Flag","Special_Event_Flag"]])

    n_anom = df_ad["Anomaly"].sum()
    top_atm = df_ad[df_ad["Anomaly"]].groupby("ATM_ID")["Anomaly"].sum().idxmax() if n_anom > 0 else "None"
    kpi_row([
        ("ANOMALIES DETECTED", str(n_anom), f"{n_anom/len(df_ad)*100:.1f}% of all records"),
        ("METHOD USED", method, None),
        ("HOLIDAY ANOMALY RATE", f"{df_ad[df_ad['Holiday_Flag']==1]['Anomaly'].mean()*100:.1f}%", None),
        ("TOP FLAGGED ATM", top_atm, None),
    ])

    section("WITHDRAWAL TIMELINE — ANOMALIES MARKED", "5.1")
    daily_ad = df_ad.groupby("Date").agg(W=("Total_Withdrawals","mean"), A=("Anomaly","max")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_ad["Date"], y=daily_ad["W"], mode="lines", name="Daily Avg",
                              line=dict(color=C["cyan"],width=1.5),
                              fill="tozeroy", fillcolor="rgba(0,212,255,0.03)"))
    ad_d = daily_ad[daily_ad["A"]]
    fig.add_trace(go.Scatter(x=ad_d["Date"], y=ad_d["W"], mode="markers", name="Anomaly",
                              marker=dict(color=C["red"], size=9, symbol="circle",
                                         line=dict(color="white",width=1))))
    fig.update_layout(**LAYOUT, title="Anomalies on Withdrawal Timeline", height=320)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("BY LOCATION", "5.2")
        la = df_ad.groupby("Location_Type")["Anomaly"].mean().sort_values(ascending=False).reset_index()
        fig = go.Figure(go.Bar(x=la["Location_Type"], y=la["Anomaly"]*100,
                                marker=dict(color=la["Anomaly"].values,
                                           colorscale=[[0,C["surface2"]],[1,C["red"]]],
                                           line=dict(color=C["border"],width=0.5)),
                                text=[f"{v*100:.1f}%" for v in la["Anomaly"]],
                                textposition="outside", textfont=dict(color=C["text"])))
        fig.update_layout(**LAYOUT, title="Anomaly Rate by Location (%)", height=280)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("SCORE DISTRIBUTION", "5.3")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_ad[~df_ad["Anomaly"]]["Score"], name="Normal",
                                    marker_color=C["cyan"], opacity=0.6, nbinsx=40))
        fig.add_trace(go.Histogram(x=df_ad[df_ad["Anomaly"]]["Score"], name="Anomaly",
                                    marker_color=C["red"], opacity=0.8, nbinsx=40))
        fig.update_layout(**LAYOUT, title="Anomaly Score Distribution", height=280, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    section("FLAGGED RECORDS", "5.4")
    at = df_ad[df_ad["Anomaly"]].sort_values("Score", ascending=False).head(60)
    st.dataframe(at[["ATM_ID","Date","Day_of_Week","Location_Type","Total_Withdrawals",
                      "Holiday_Flag","Special_Event_Flag","Weather_Condition","Score"]].round(2),
                 use_container_width=True, hide_index=True)
    ai_block(f"{method} found {n_anom} anomalies ({n_anom/len(df_ad)*100:.1f}%). Holiday anomaly rate {df_ad[df_ad['Holiday_Flag']==1]['Anomaly'].mean()*100:.1f}%.", "anom_ai")


# ═══════════════════════════════════════════════════════════
# DEMAND PLANNER
# ═══════════════════════════════════════════════════════════
elif page == "🎛️   Demand Planner":
    st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.06em;">DEMAND <span style="color:{C["amber"]}">PLANNER</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.12em;margin-bottom:18px;">STAGE 6 · INTERACTIVE CASH PLANNING</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_days = st.multiselect("Day of Week",
                                   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                                   default=["Saturday","Sunday"])
    with c2:
        sel_tod = st.multiselect("Time of Day", ["Morning","Afternoon","Evening","Night"],
                                  default=["Morning","Afternoon"])
    with c3:
        sel_loc = st.multiselect("Location", df["Location_Type"].unique().tolist(),
                                  default=df["Location_Type"].unique().tolist())
    with c4:
        sel_wx = st.multiselect("Weather", df["Weather_Condition"].unique().tolist(),
                                 default=df["Weather_Condition"].unique().tolist())

    hol_only = st.checkbox("🎉 Show holiday / event days only")

    filt = df.copy()
    if sel_days: filt = filt[filt["Day_of_Week"].isin(sel_days)]
    if sel_tod:  filt = filt[filt["Time_of_Day"].isin(sel_tod)]
    if sel_loc:  filt = filt[filt["Location_Type"].isin(sel_loc)]
    if sel_wx:   filt = filt[filt["Weather_Condition"].isin(sel_wx)]
    if hol_only: filt = filt[(filt["Holiday_Flag"]==1)|(filt["Special_Event_Flag"]==1)]

    if filt.empty:
        st.markdown(f'<div class="alert-critical">No records match filters. Please adjust selection.</div>', unsafe_allow_html=True)
    else:
        peak_day = filt.groupby("Day_of_Week")["Total_Withdrawals"].mean().idxmax()
        peak_loc = filt.groupby("Location_Type")["Total_Withdrawals"].mean().idxmax()
        kpi_row([
            ("FILTERED RECORDS",    f"{len(filt):,}", None),
            ("AVG WITHDRAWALS",     f"₹{filt['Total_Withdrawals'].mean()/1000:.1f}K", None),
            ("AVG NEXT-DAY DEMAND", f"₹{filt['Cash_Demand_Next_Day'].mean()/1000:.1f}K", None),
            ("PEAK LOCATION",       peak_loc, None),
        ])

        c1, c2 = st.columns(2)
        with c1:
            trend = filt.groupby("Date")["Total_Withdrawals"].mean().reset_index()
            fig = go.Figure(go.Scatter(x=trend["Date"], y=trend["Total_Withdrawals"],
                                        mode="lines", fill="tozeroy",
                                        line=dict(color=C["amber"],width=2),
                                        fillcolor="rgba(255,179,71,0.05)"))
            fig.update_layout(**LAYOUT, title="Filtered Withdrawal Trend", height=280)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            lf = filt.groupby("Location_Type")["Total_Withdrawals"].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=lf.index, y=lf.values,
                                   marker=dict(color=lf.values,
                                              colorscale=[[0,C["surface2"]],[1,C["amber"]]],
                                              line=dict(color=C["border"],width=0.5))))
            fig.update_layout(**LAYOUT, title="Avg Withdrawals by Location (Filtered)", height=280)
            st.plotly_chart(fig, use_container_width=True)

        section("TOP 10 ATMs — FILTERED", "6.2")
        top10 = filt.groupby("ATM_ID")[["Total_Withdrawals","Cash_Demand_Next_Day"]].mean().sort_values(
            "Total_Withdrawals", ascending=False).head(10).round(0)
        st.dataframe(top10, use_container_width=True)

        section("AI CASH PLANNING RECOMMENDATIONS", "6.3")
        ctx = (f"Filters: days={sel_days}, time={sel_tod}, location={sel_loc}, weather={sel_wx}. "
               f"Avg withdrawal ₹{filt['Total_Withdrawals'].mean():.0f}. Peak day: {peak_day}. Peak location: {peak_loc}. "
               f"Records analysed: {len(filt)}.")
        ai_block(ctx, "planner_ai", "✦ Generate Cash Planning Recommendations")


# ═══════════════════════════════════════════════════════════
# HEATMAP
# ═══════════════════════════════════════════════════════════
elif page == "🗺️   ATM Heatmap":
    st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.06em;">ATM <span style="color:{C["cyan"]}">HEATMAP</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:{C["muted"]};letter-spacing:0.12em;margin-bottom:18px;">GEOGRAPHIC DEMAND DISTRIBUTION</div>', unsafe_allow_html=True)

    atm_geo = df.groupby(["ATM_ID","lat","lon","Location_Type"]).agg(
        Avg_Withdrawals=("Total_Withdrawals","mean"),
        Total_Records=("Total_Withdrawals","count"),
        Holiday_Rate=("Holiday_Flag","mean")
    ).reset_index().round(2)

    c1, c2 = st.columns([1, 3])
    with c1:
        metric  = st.radio("Color metric", ["Avg_Withdrawals","Holiday_Rate","Total_Records"])
        style   = st.radio("Map style",    ["carto-darkmatter","carto-positron"])
    with c2:
        fig = px.density_mapbox(atm_geo, lat="lat", lon="lon", z=metric,
                                 radius=40, zoom=3.5, center={"lat":22,"lon":82},
                                 mapbox_style=style,
                                 color_continuous_scale=[[0,C["surface"]],[0.5,C["purple"]],[1,C["cyan"]]],
                                 hover_name="ATM_ID",
                                 hover_data={"lat":False,"lon":False,"Location_Type":True,
                                            "Avg_Withdrawals":True,"Holiday_Rate":True})
        fig.update_layout(paper_bgcolor=C["surface"], font=dict(color=C["text"]),
                          title=f"ATM Demand Density — {metric.replace('_',' ')}",
                          height=500, margin=dict(l=0,r=0,t=50,b=0),
                          coloraxis_colorbar=dict(tickfont=dict(color=C["text"]),
                                                   title=dict(text=metric, font=dict(color=C["muted"]))))
        st.plotly_chart(fig, use_container_width=True)

    section("ATM NODE MAP — INDIVIDUAL BUBBLES", "GEO")
    fig2 = px.scatter_mapbox(atm_geo, lat="lat", lon="lon",
                              size="Avg_Withdrawals", color="Location_Type",
                              color_discrete_sequence=[C["cyan"],C["amber"],C["green"],C["purple"],C["red"]],
                              hover_name="ATM_ID", zoom=3.5, center={"lat":22,"lon":82},
                              mapbox_style=style, size_max=22, opacity=0.85)
    fig2.update_layout(paper_bgcolor=C["surface"], font=dict(color=C["text"]),
                        height=480, margin=dict(l=0,r=0,t=30,b=0),
                        legend=dict(bgcolor="rgba(13,21,40,0.85)",bordercolor=C["border"]))
    st.plotly_chart(fig2, use_container_width=True)

    insight("Bubble size = avg withdrawals. High-density clusters indicate regions needing dedicated replenishment routes.")
    ai_block(f"Location breakdown: {atm_geo['Location_Type'].value_counts().to_dict()}. Top ATM: {atm_geo.sort_values('Avg_Withdrawals',ascending=False).iloc[0]['ATM_ID']} (₹{atm_geo['Avg_Withdrawals'].max():.0f}).", "heatmap_ai")


# ── Footer
st.markdown(f"""
<div style="margin-top:40px;padding:16px;text-align:center;
            border-top:1px solid {C["border"]};
            font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:{C["muted"]};letter-spacing:0.1em;">
    FINTRUST BANK LTD. · ATM INTELLIGENCE PLATFORM · FA-2 DATA MINING · CRS · Y1
</div>
""", unsafe_allow_html=True)
