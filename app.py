import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & ARCHITECTURE THEME
# ==========================================
st.set_page_config(
    page_title="Sales Predictive Studio Pro",
    page_icon="📊",
    layout="wide"  # Deploys wide layout for high-density dashboard matrix scanning
)

# Premium Custom Glassmorphism Theme Overlays
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Reset base typography styles */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #07090e;
    color: #f3f4f6;
}

/* Glassmorphism Premium Container Grid */
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6);
    margin-bottom: 20px;
}

/* Premium KPI Value Layout Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    text-align: center;
}

.kpi-value {
    font-size: 32px;
    font-weight: 900;
    margin-top: 5px;
}

.text-neon-blue { color: #00f5ff; text-shadow: 0 0 10px rgba(0,245,255,0.3); }
.text-neon-green { color: #10b981; text-shadow: 0 0 10px rgba(16,185,129,0.3); }
.text-neon-orange { color: #f59e0b; text-shadow: 0 0 10px rgba(245,158,11,0.3); }

/* Analytical Tags Matrix */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 6px;
}
.badge-blue { background: rgba(0, 122, 255, 0.15); color: #007aff; border: 1px solid rgba(0, 122, 255, 0.3); }
.badge-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Animation framework */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.animated-view {
    animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# RUNTIME INITIALIZATION RUN-LOGS
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = [
        {"Timestamp": "10:14:22", "Baseline ($)": 14200, "Scenario": "Standard Baseline Optimization", "Yield ($)": 15843.50, "Status": "🟢 Expansion"},
        {"Timestamp": "11:05:40", "Baseline ($)": 15600, "Scenario": "Q4 Peak Holiday Surge", "Yield ($)": 24155.80, "Status": "🔥 Peak Surge"},
    ]

# ==========================================
# SIDEBAR CONTROL DECK FRAMEWORKS
# ==========================================
st.sidebar.markdown("## 🕹️ Simulation Parameters")
macro_scenario = st.sidebar.selectbox(
    "🌐 Market Macro Scenario", 
    ["Standard Baseline Optimization", "Q4 Peak Holiday Surge", "Market Contraction Recession"]
)

st.sidebar.markdown("---")
marketing_spend = st.sidebar.slider("🚀 Ads Campaign Vector Budget ($)", 0, 20000, 3500, step=500)
selected_day = st.sidebar.selectbox("📆 Target Projection Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

st.sidebar.markdown("---")
# New Feature addition: Competitor Pricing Shock Option Parameter
competitor_shock = st.sidebar.checkbox("🚨 Apply Competitor Aggressive Price Cuts")

# ==========================================
# APPLICATION BODY MAIN CORE INTERFACE
# ==========================================
st.markdown("<h1 style='text-align: center; font-weight:900; margin-bottom:0;'>📊 Sales Predictive Studio Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b7280; font-size:14px; margin-bottom:25px;'>Production-Grade Time-Series Modeling Engine & Continuous Decision Telemetry Dashboard</p>", unsafe_allow_html=True)

# Main Grid Container System
layout_col1, layout_col2 = st.columns([1, 2])

with layout_col1:
    st.markdown("<div class='glass-panel' style='height: 100%;'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Time-Series Metrics")
    
    historical_baseline = st.number_input("📈 Previous Week Total Volume ($):", min_value=100, max_value=1000000, value=14500, step=500)
    rolling_mean_7d = st.number_input("🔄 7-Day Rolling Operational Mean ($):", min_value=100, max_value=1000000, value=13800, step=500)
    seasonal_lag_30d = st.number_input("📅 30-Day Cyclical Month Lag ($):", min_value=100, max_value=1000000, value=15200, step=500)
    
    forecast_btn = st.button("⚡ Run Live Predictive Pipeline", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# DETERMINISTIC DEEP ALGEBRAIC MODEL MATRIX ENGINE
# ==========================================
def calculate_horizon_predictions(hist, roll, lag, start_day, ads, scenario, shock):
    day_indices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}
    
    # Extract target macro matrix multipliers
    scenario_scalars = {"Standard Baseline Optimization": 1.00, "Q4 Peak Holiday Surge": 1.42, "Market Contraction Recession": 0.71}
    current_scalar = scenario_scalars[scenario]
    
    # Calculate core model feature weight vectors
    feature_mix_base = (hist * 0.25) + (roll * 0.45) + (lag * 0.30)
    ad_boost_coefficient = float(ads) * 0.88
    
    # Competitor shock scalar parameter checks
    shock_scalar = 0.82 if shock else 1.00
    
    start_idx = day_indices.index(start_day)
    ordered_days = day_indices[start_idx:] + day_indices[:start_idx]
    
    forecast_timeline_curve = []
    decay_factor = 1.0
    
    for i, day in enumerate(ordered_days):
        # Apply structural rolling horizon decay multipliers
        decay_factor -= (i * 0.012)
        day_yield = (feature_mix_base + ad_boost_coefficient) * day_weights[day] * current_scalar * shock_scalar * decay_factor
        forecast_timeline_curve.append((day, float(day_yield)))
        
    return forecast_timeline_curve

# ==========================================
# MAIN DASHBOARD DYNAMIC GRAPH AND KPI ENGINE OUTPUT
# ==========================================
if layout_col2.markdown("<div class='animated-view'>", unsafe_allow_html=True) or True:
    
    # Compute baseline variables immediately to display active telemetry states
    forecast_curve = calculate_horizon_predictions(
        historical_baseline, rolling_mean_7d, seasonal_lag_30d,
        selected_day, marketing_spend, macro_scenario, competitor_shock
    )
    
    target_day_yield = forecast_curve[0][1]
    
    # 1. METRICS VALUATION KPI ROW PANELS
    with layout_col2:
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            variance_pct = ((target_day_yield - rolling_mean_7d) / rolling_mean_7d) * 100
            neon_class = "text-neon-green" if variance_pct >= 0 else "text-neon-orange"
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='font-size:11px; text-transform:uppercase; color:#8e8e93;'>Projected Target Yield</div>
                <div class='kpi-value {neon_class}'>${target_day_yield:,.2f}</div>
                <div style='font-size:11px; margin-top:4px; color:{"#10b981" if variance_pct >= 0 else "#ef4444"}'>
                    {"▲" if variance_pct >= 0 else "▼"} {variance_pct:.1f}% vs 7D Avg
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='font-size:11px; text-transform:uppercase; color:#8e8e93;'>7-Day Volume Track</div>
                <div class='kpi-value text-neon-blue'>${sum([v for _, v in forecast_curve]):,.2f}</div>
                <div style='font-size:11px; margin-top:4px; color:#8e8e93;'>Cumulative Horizon Total</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            # Active status badge logic check
            if macro_scenario == "Q4 Peak Holiday Surge":
                badge_html = "<span class='badge badge-green'>🔥 PEAK SURGE ACTIVE</span>"
            elif macro_scenario == "Market Contraction Recession" or competitor_shock:
                badge_html = "<span class='badge badge-red'>⚠️ HIGH VOLATILITY</span>"
            else:
                badge_html = "<span class='badge badge-blue'>🟢 STABLE OPERATIONS</span>"
                
            st.markdown(f"""
            <div class='kpi-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                <div style='font-size:11px; text-transform:uppercase; color:#8e8e93; margin-bottom:5px;'>System Strategic Vector</div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        # 2. RENDER THE HIGH-PERFORMANCE 7-DAY TIMELINE CHART
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        days_axis = [d for d, _ in forecast_curve]
        yields_axis = [v for _, v in forecast_curve]
        
        fig_timeline = go.Figure()
        
        # Add the main gradient-shaded area line trace curve
        fig_timeline.add_trace(go.Scatter(
            x=days_axis, y=yields_axis,
            mode='lines+markers',
            line=dict(color='#00f5ff', width=3.5),
            marker=dict(size=8, color='#007aff', line=dict(color='#fff', width=1)),
            fill='tozeroy',
            fillcolor='rgba(0, 245, 255, 0.03)'
        ))
        
        # Overlay the historical 7D reference line path parameters
        fig_timeline.add_trace(go.Scatter(
            x=days_axis, y=[rolling_mean_7d]*7,
            mode='lines',
            name='7D Mean Reference Line',
            line=dict(color='rgba(255,255,255,0.2)', width=2, dash='dash')
        ))
        
        fig_timeline.update_layout(
            title="🎯 7-Day Progressive Target Predictive Velocity Horizon Path",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#8e8e93",
            title_font=dict(size=14, color="#fff", family="Inter"),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff')),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', title="Financial Metrics Valuation ($)", titlefont=dict(size=11)),
            margin=dict(l=10, r=10, t=45, b=10),
            height=280,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_timeline, width='stretch')

# ==================== SECONDARY INTERACTIVE SUB-GRID RECONCILIATIONS ====================
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
sub_col1, sub_col2 = st.columns([1, 1])

with sub_col1:
    st.markdown("<div class='glass-panel' style='height:100%;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Calculated Financial Distribution Matrix")
    
    # Render composition metrics map
    pie_labels = ["Base Core Target Matrix", "Campaign Budget Boost", "Macro Scaled Weight Variance"]
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
    st.markdown("### 🎞️ Enterprise Session Audit History Log")
    
    # Handle instant appending log logic chains
    if forecast_btn:
        st.session_state.history.append({
            "Timestamp": time.strftime("%H:%M:%S"),
            "Baseline ($)": int(historical_baseline),
            "Scenario": macro_scenario if not competitor_shock else f"🚨 Shock: {macro_scenario[:12]}...",
            "Yield ($)": round(target_day_yield, 2),
            "Status": "🔥 Peak Surge" if target_day_yield > historical_baseline * 1.1 else "🟢 Expansion" if target_day_yield >= historical_baseline else "🔴 Contraction"
        })
        
    df_history = pd.DataFrame(st.session_state.history[::-1]) # Reverse order to display newest runs on top rows
    st.dataframe(df_history, width='stretch', height=140)
    
    if st.button("🗑️ Reset Audit Telemetry Databases", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER CLOSING CHECKS ====================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 11px;'>🍿 Enterprise Decision Infrastructure Ecosystem Deployed via Glassmorphic Multi-Layer Vector Cloud Ensembles</div>", unsafe_allow_html=True)
