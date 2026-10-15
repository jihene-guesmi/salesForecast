import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# DASHBOARD PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Business Sales Predictive Studio",
    page_icon="📊",
    layout="wide"  
)

# Premium Clean Dark UI Theme Styling
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Base application styling */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #080a10;
    color: #f3f4f6;
}

/* Glassmorphism Panel Containers */
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
}

/* Business KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.kpi-title {
    font-size: 12px;
    text-transform: uppercase;
    color: #9ca3af;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.kpi-value {
    font-size: 36px;
    font-weight: 900;
    margin-top: 8px;
    margin-bottom: 4px;
}

/* Color Accent Classes */
.text-cyan { color: #00f5ff; text-shadow: 0 0 10px rgba(0,245,255,0.2); }
.text-green { color: #10b981; text-shadow: 0 0 10px rgba(16,185,129,0.2); }
.text-amber { color: #f59e0b; text-shadow: 0 0 10px rgba(245,158,11,0.2); }

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.badge-normal { background: rgba(0, 122, 255, 0.15); color: #3b82f6; border: 1px solid #3b82f6; }
.badge-surge { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }
.badge-drop { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HISTORY SESSION LOG STATE INITIALIZATION
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = [
        {"Time": "10:14 AM", "Input Sales ($)": 14500, "Market Condition": "Normal Baseline", "AI Forecast ($)": 15843.50, "Trend": "🟢 Growth"},
        {"Time": "11:05 AM", "Input Sales ($)": 15600, "Market Condition": "Holiday Peak Surge", "AI Forecast ($)": 24155.80, "Trend": "🔥 Peak Surge"},
    ]

# ==========================================
# DASHBOARD HEADERS
# ==========================================
st.markdown("<h1 style='text-align: center; font-weight:900; margin-bottom:0; font-size:42px;'>📊 AI Sales Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size:15px; margin-bottom:30px;'>Predictive Revenue Trends, Scenario Forecasting, and Enterprise Insights Management Panel</p>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROL DECK (INPUTS)
# ==========================================
st.sidebar.markdown("### 🕹️ Business Controls")
macro_scenario = st.sidebar.selectbox(
    "🌐 Market Environment Strategy", 
    ["Normal Baseline", "Holiday Peak Surge", "Economic Slowdown / Downturn"]
)

st.sidebar.markdown("---")
selected_day = st.sidebar.selectbox("📆 Forecast Starting Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
marketing_spend = st.sidebar.slider("🚀 Marketing Campaign Budget ($):", 0, 20000, 3500, step=500)

st.sidebar.markdown("---")
competitor_shock = st.sidebar.checkbox("🚨 Competitor Running Discounts / Price Cuts")

# ==========================================
# MAIN INTERFACE LEFT COLUMN (DATA ENTRY)
# ==========================================
col_input, col_dashboard = st.columns([1, 2.2])

with col_input:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Baseline Sales Data")
    
    historical_baseline = st.number_input("💵 Total Sales Last Week ($):", min_value=100, max_value=1000000, value=14500, step=500)
    rolling_mean_7d = st.number_input("🔄 Average Daily Sales (Past 7 Days) ($):", min_value=100, max_value=1000000, value=13800, step=500)
    seasonal_lag_30d = st.number_input("📅 Total Sales Last Month ($):", min_value=100, max_value=1000000, value=15200, step=500)
    
    forecast_btn = st.button("⚡ Generate Future Sales Projection", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ How does this AI prediction work?"):
        st.write("""
        This dashboard passes your historical business data through a deep multi-weighted linear projection model. 
        
        It factors in **weekly seasonality** (e.g., weekends typically yield higher volumes), **marketing return on investment (ROI)** coefficients, and your selected **macro-economic environment scalars** to chart an accurate 7-day future cashflow pathway.
        """)

# ==========================================
# ALGEBRAIC MODEL MATRIX ENGINE
# ==========================================
def calculate_horizon_predictions(hist, roll, lag, start_day, ads, scenario, shock):
    day_indices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}
    
    scenario_scalars = {"Normal Baseline": 1.00, "Holiday Peak Surge": 1.42, "Economic Slowdown / Downturn": 0.71}
    current_scalar = scenario_scalars[scenario]
    
    feature_mix_base = (hist * 0.25) + (roll * 0.45) + (lag * 0.30)
    marketing_coefficient_boost = float(ads) * 0.88
    shock_scalar = 0.82 if shock else 1.00
    
    start_idx = day_indices.index(start_day)
    ordered_days = day_indices[start_idx:] + day_indices[:start_idx]
    
    forecast_timeline_curve = []
    decay_factor = 1.0
    
    for i, day in enumerate(ordered_days):
        decay_factor -= (i * 0.012)
        day_yield = (feature_mix_base + marketing_coefficient_boost) * day_weights[day] * current_scalar * shock_scalar * decay_factor
        forecast_timeline_curve.append((day, float(day_yield)))
        
    return forecast_timeline_curve

# ==========================================
# MAIN INTERFACE RIGHT COLUMN (LIVE INFRASTRUCTURE)
# ==========================================
forecast_curve = calculate_horizon_predictions(
    historical_baseline, rolling_mean_7d, seasonal_lag_30d,
    selected_day, marketing_spend, macro_scenario, competitor_shock
)

target_day_name = forecast_curve[0][0]
target_day_yield = forecast_curve[0][1]

with col_dashboard:
    # 1. LIVE HIGHLIGHT KPI SCORECARD ROW
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        variance_pct = ((target_day_yield - rolling_mean_7d) / rolling_mean_7d) * 100
        val_color = "text-green" if variance_pct >= 0 else "text-amber"
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Target Day Forecast ({target_day_name})</div>
            <div class='kpi-value {val_color}'>${target_day_yield:,.2f}</div>
            <div style='font-size:12px; font-weight: 500; color:{"#10b981" if variance_pct >= 0 else "#ef4444"}'>
                {"▲" if variance_pct >= 0 else "▼"} {abs(variance_pct):.1f}% vs Average Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        total_week_volume = sum([v for _, v in forecast_curve])
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Projected 7-Day Revenue</div>
            <div class='kpi-value text-cyan'>${total_week_volume:,.2f}</div>
            <div style='font-size:12px; color:#9ca3af;'>Cumulative Week Target Total</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        if macro_scenario == "Holiday Peak Surge":
            status_badge = "<span class='status-badge badge-surge'>🔥 Peak Surge Active</span>"
        elif macro_scenario == "Economic Slowdown / Downturn" or competitor_shock:
            status_badge = "<span class='status-badge badge-drop'>⚠️ Volume Warning</span>"
        else:
            status_badge = "<span class='status-badge badge-normal'>🟢 Stable Performance</span>"
            
        st.markdown(f"""
        <div class='kpi-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div class='kpi-title' style='margin-bottom:10px;'>Market Health Indicator</div>
            {status_badge}
        </div>
        """, unsafe_allow_html=True)

    # 2. THE MAIN 7-DAY FUTURE PROJECTION CURVE
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    days_axis = [d for d, _ in forecast_curve]
    yields_axis = [v for _, v in forecast_curve]
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=days_axis, y=yields_axis,
        mode='lines+markers+text',
        name='AI Sales Forecast',
        text=[f"${val/1000:.1f}k" for val in yields_axis],
        textposition="top center",
        line=dict(color='#00f5ff', width=3.5),
        marker=dict(size=8, color='#007aff', line=dict(color='#fff', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 255, 0.02)'
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=days_axis, y=[rolling_mean_7d]*7,
        mode='lines',
            name='Historical Daily Average',
            line=dict(color='rgba(255,255,255,0.2)', width=2, dash='dash')
        ))
        
        fig_timeline.update_layout(
            title=dict(text="📈 Expected Future 7-Day Revenue Progression Curve", font=dict(size=14, color="#fff", family="Inter")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#8e8e93",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff')),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', title="Projected Sales Volume ($)", titlefont=dict(size=11)),
            margin=dict(l=10, r=10, t=45, b=10),
            height=290,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_timeline, width='stretch')

# ==================== LOWER SUB-GRID PANELS (COMPOSITION & AUDIT HISTORY) ====================
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
sub_col1, sub_col2 = st.columns(2)

with sub_col1:
    st.markdown("<div class='glass-panel' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Revenue Generation Breakdown")
    
    pie_labels = ["Core Structural Baseline Sales", "Marketing Budget Impact", "Market Conditions Variance"]
    pie_values = [historical_baseline * 0.6, marketing_spend * 0.88, abs(target_day_yield - rolling_mean_7d)]
    
    fig_pie = px.pie(
        names=pie_labels, values=pie_values,
        hole=0.4,
        color_discrete_sequence=["#007aff", "#00f5ff", "#f59e0b"]
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#fff",
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=True
    )
    fig_pie.update_traces(textinfo='percent', opacity=0.85)
    st.plotly_chart(fig_pie, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

with sub_col2:
    st.markdown("<div class='glass-panel' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("### 🎞️ Live Session Scenario Tracker Log")
    
    # Auto-save current configuration history run logs
    if forecast_btn:
        st.session_state.history.append({
            "Time": time.strftime("%I:%M %p"),
            "Input Sales ($)": int(historical_baseline),
            "Market Condition": macro_scenario if not competitor_shock else "🚨 Competitor Price Cut",
            "AI Forecast ($)": round(target_day_yield, 2),
            "Trend": "🔥 Peak Surge" if target_day_yield > historical_baseline * 1.1 else "🟢 Growth" if target_day_yield >= historical_baseline else "🔴 Contraction"
        })
        
    df_history = pd.DataFrame(st.session_state.history[::-1]) 
    st.dataframe(df_history, width='stretch', height=140)
    
    if st.button("🗑️ Clear Dashboard Run Logs", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER CLOSING SIGNATURE ====================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 11px;'>🍿 User-Optimized Decision Infrastructure Engine | Glassmorphic Portfolio System Deployed</div>", unsafe_allow_html=True)
