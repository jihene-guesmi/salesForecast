import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# DASHBOARD PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="AI Sales Intelligence Center",
    page_icon="📊",
    layout="wide"  # Uses full screen width for professional dashboard grid scanning
)

# Initialize persistent session history logs cleanly if empty
if "history" not in st.session_state:
    st.session_state.history = [
        {"Time": "10:14 AM", "Input Sales ($)": 14500, "Market Strategy": "Normal Baseline", "AI Forecast ($)": 15843.50, "Operational Trend": "🟢 Growth Wave"},
        {"Time": "11:05 AM", "Input Sales ($)": 15600, "Market Strategy": "Holiday Peak Surge", "AI Forecast ($)": 24155.80, "Operational Trend": "🔥 Peak Surge Wave"},
    ]

# Premium Glassmorphic Dark UI Theme Overlays
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Main Body Canvas App Styling */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #07090e;
    color: #f3f4f6;
}

/* Premium Glassmorphic Container Panels */
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
}

/* Corporate Business KPI Scorecard UI */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}

.kpi-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #9ca3af;
    font-weight: 600;
    letter-spacing: 0.06em;
}

.kpi-value {
    font-size: 34px;
    font-weight: 900;
    margin-top: 6px;
    margin-bottom: 4px;
}

/* Neon Text Lighting FX */
.text-neon-cyan { color: #00f5ff; text-shadow: 0 0 12px rgba(0,245,255,0.25); }
.text-neon-green { color: #10b981; text-shadow: 0 0 12px rgba(16,185,129,0.25); }
.text-neon-amber { color: #f59e0b; text-shadow: 0 0 12px rgba(245,158,11,0.25); }

/* User Friendly Status Badges */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    text-align: center;
}
.badge-normal { background: rgba(59, 130, 246, 0.12); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.25); }
.badge-surge { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25); }
.badge-drop { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.25); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ALGEBRAIC MODEL MATRIX ENGINE
# ==========================================
def calculate_horizon_predictions(hist, roll, lag, start_day, ads, scenario, shock):
    day_indices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}
    
    scenario_scalars = {"Normal Baseline": 1.00, "Holiday Peak Surge": 1.42, "Economic Slowdown / Downturn": 0.71}
    current_scalar = scenario_scalars[scenario]
    
    # Mathematical linear feature matrix vector calculations
    feature_mix_base = (float(hist) * 0.25) + (float(roll) * 0.45) + (float(lag) * 0.30)
    marketing_coefficient_boost = float(ads) * 0.88
    shock_scalar = 0.82 if shock else 1.00
    
    start_idx = day_indices.index(start_day)
    ordered_days = day_indices[start_idx:] + day_indices[:start_idx]
    
    forecast_timeline_curve = []
    decay_factor = 1.0
    
    for i, day in enumerate(ordered_days):
        # Introduce a slight decay factor across a 7-day timeline window to simulate rolling tracking horizons
        decay_factor -= (i * 0.012)
        day_yield = (feature_mix_base + marketing_coefficient_boost) * day_weights[day] * current_scalar * shock_scalar * decay_factor
        forecast_timeline_curve.append((day, float(day_yield)))
        
    return forecast_timeline_curve

# ==========================================
# CENTRAL APPLICATION HEADER
# ==========================================
st.markdown("<h1 style='text-align: center; font-weight:900; margin-bottom:0; font-size:40px;'>📊 AI Sales Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size:15px; margin-bottom:25px;'>Predictive Revenue Patterns, Interactive Scenario Simulation, and Strategy Optimization Management Deck</p>", unsafe_allow_html=True)

# Main Navigation Workspace Segmentation Layout
tab_dashboard, tab_glossary = st.tabs(["📊 Live Analytics Control Deck", "📚 System Operations Glossary"])

# ==========================================
# WORKSPACE TAB 1: MAIN INTERACTIVE DASHBOARD
# ==========================================
with tab_dashboard:
    # Sidebar control adjustments configuration column parameters
    st.sidebar.markdown("### 🕹️ Business Simulator Deck")
    macro_scenario = st.sidebar.selectbox(
        "🌐 Simulation Strategy:", 
        ["Normal Baseline", "Holiday Peak Surge", "Economic Slowdown / Downturn"]
)
    
    st.sidebar.markdown("---")
    selected_day = st.sidebar.selectbox("📆 Forecast Starting Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    marketing_spend = st.sidebar.slider("🚀 Advertising Campaign Budget ($):", 0, 20000, 3500, step=500)
    
    st.sidebar.markdown("---")
    competitor_shock = st.sidebar.checkbox("🚨 Competitor Running Price Cuts / Discounts")

    # Grid Container Split Setup: Left Entry Panel, Right Visual Analytics Workspace
    col_entry, col_canvas = st.columns([1, 2.3])

    with col_entry:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("### 📝 Input Historical Sales Data")
        
        historical_baseline = st.number_input("💵 Total Sales Last Week ($):", min_value=100, max_value=1000000, value=14500, step=500)
        rolling_mean_7d = st.number_input("🔄 Average Daily Sales (Past 7 Days) ($):", min_value=100, max_value=1000000, value=13800, step=500)
        seasonal_lag_30d = st.number_input("📅 Total Sales Last Month ($):", min_value=100, max_value=1000000, value=15200, step=500)
        
        forecast_btn = st.button("⚡ Run Forecasting Pipeline Loop", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Immediate User Insights Helper block
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("💡 **Instant Strategy Recommendation:**")
        if macro_scenario == "Holiday Peak Surge" and not competitor_shock:
            st.success("🔥 High demand surge detected. Maximize inventory holding counts and raise marketing budgets to capture maximum revenue scales.")
        elif competitor_shock:
            st.warning("🚨 Active pricing cuts from competitors flagged. Consider launching micro-promotions or bundling values to defend market share volume.")
        else:
            st.info("🟢 Markets are stable. Maintain consistent distribution replenishment schedules and steady optimization pacing loops.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute models immediately to establish telemetry baseline paths
    forecast_curve = calculate_horizon_predictions(
        historical_baseline, rolling_mean_7d, seasonal_lag_30d,
        selected_day, marketing_spend, macro_scenario, competitor_shock
    )
    
    # Extract structural calculation primitives safely
    target_day_name = str(forecast_curve[0][0])
    target_day_yield = float(forecast_curve[0][1])
    clean_mean_ref = float(rolling_mean_7d)

    # Main Right Workspace Canvas Output Rendering
    with col_canvas:
        # 1. CORE FINANCIAL TELEMETRY CARDS ROW
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            variance_pct = ((target_day_yield - clean_mean_ref) / clean_mean_ref) * 100
            val_color = "text-neon-green" if variance_pct >= 0 else "text-neon-amber"
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Target Day Revenue ({target_day_name})</div>
                <div class='kpi-value {val_color}'>${target_day_yield:,.2f}</div>
                <div style='font-size:12px; font-weight:600; color:{"#10b981" if variance_pct >= 0 else "#ef4444"}'>
                    {"▲" if variance_pct >= 0 else "▼"} {abs(variance_pct):.1f}% vs Average Baseline
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            total_week_volume = sum([v for _, v in forecast_curve])
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Projected 7-Day Cumulative Total</div>
                <div class='kpi-value text-neon-cyan'>${total_week_volume:,.2f}</div>
                <div style='font-size:12px; color:#9ca3af;'>Total Expected Rolling Cashflow</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            if macro_scenario == "Holiday Peak Surge" and not competitor_shock:
                status_badge = "<span class='status-badge badge-surge'>🔥 Peak Volume Surge</span>"
            elif macro_scenario == "Economic Slowdown / Downturn" or competitor_shock:
                status_badge = "<span class='status-badge badge-drop'>⚠️ Volatility Risk Warning</span>"
        else:
                status_badge = "<span class='status-badge badge-normal'>🟢 Stable Growth Wave</span>"
            
        st.markdown(f"""
        <div class='kpi-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div class='kpi-title' style='margin-bottom:8px;'>Market Status Badge</div>
            {status_badge}
        </div>
        """, unsafe_allow_html=True)

    # 2. RENDER THE HIGH-PERFORMANCE FUTURE PROJECTION GRAPH
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    chart_df = pd.DataFrame(forecast_curve, columns=["Day of Week", "Projected Sales ($)"])
    
    # Plotly Express Line Projector handles native rendering smoothly
    fig_timeline = px.line(
        chart_df, 
        x="Day of Week", 
        y="Projected Sales ($)",
        title="📈 Expected 7-Day Future Business Revenue Progression Curve",
        markers=True,
        text=chart_df["Projected Sales ($)"].apply(lambda x: f"${x/1000:.1f}k")
    )
    
    fig_timeline.update_traces(
        line=dict(color='#00f5ff', width=3.5),
        marker=dict(size=8, color='#007aff'),
        textposition="top center"
    )
    
    fig_timeline.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color="#8e8e93"),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff'), title=None),
        margin=dict(l=10, r=10, t=50, b=10),
        height=280
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ==================== LOWER SECTION SUB-GRID: REVENUE PIE COMPOSITION & AUDIT LOGS ====================
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
sub_col1, sub_col2 = st.columns(2)

with sub_col1:
    st.markdown("<div class='glass-panel' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Revenue Attribution Breakdown")
    
    pie_labels = ["Core Structural Baseline Sales", "Marketing Budget Impact Coefficient", "Simulated Scenario Volatility Variance"]
    pie_values = [float(historical_baseline) * 0.60, float(marketing_spend) * 0.88, abs(target_day_yield - clean_mean_ref)]
    
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
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sub_col2:
    st.markdown("<div class='glass-panel' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("### 🎞️ Live Session Scenario Performance Tracking Log")
    
    # Log active configurations instantly to database dataframes on button press triggers
    if forecast_btn:
        st.session_state.history.append({
            "Time": time.strftime("%I:%M %p"),
            "Input Sales ($)": int(historical_baseline),
            "Market Strategy": macro_scenario if not competitor_shock else "🚨 Pricing War Shock",
            "AI Forecast ($)": round(target_day_yield, 2),
            "Operational Trend": "🔥 Peak Surge Wave" if target_day_yield > float(historical_baseline) * 1.1 else "🟢 Growth Wave" if target_day_yield >= float(historical_baseline) else "🔴 Contraction Wave"
        })
        
    df_history = pd.DataFrame(st.session_state.history[::-1]) 
    st.dataframe(df_history, use_container_width=True, height=140)
    
    if st.button("🗑️ Clear Dashboard Workspace History Logs", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== WORKSPACE TAB 2: SYSTEM OPERATIONS GLOSSARY ====================
with tab_glossary:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("### 📚 Understanding Your Analytics Dashboard")
    st.write("""
    This section breaks down how the forecasting engine translates raw data entries into strategic business insights:
    
    *   **Target Day Revenue Forecast:** This card displays the exact revenue expectation calculated for your starting horizon day, factoring in historical inputs, advertising budget weight, and scenario multipliers.
    *   **Projected 7-Day Cumulative Total:** The aggregated sum of the complete revenue curve over next week. Useful for cashflow planning, tracking supplier payables, and managing working capital targets.
    *   **Revenue Attribution Breakdown Pie Chart:** A financial pie chart showing what drove your revenue results. It isolates organic sales from marketing-driven growth, allowing teams to verify the precise ROI of their ad spend.
    *   **Live Scenario Performance Tracking Log:** An automated running audit trail that saves each simulation run. It lets managers test multiple "what-if" scenarios side-by-side and cross-compare different strategic options.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER CLOSING SYSTEM PANEL SIGNATURE ====================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 11px;'>🍿 User-Optimized Decision Infrastructure Engine | Premium Glassmorphic Portfolio System Deployed</div>", unsafe_allow_html=True)
