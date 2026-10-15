import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & PREMIUM BLACK THEME
# ==========================================
st.set_page_config(
    page_title="Predictive Sales Studio Pro",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://googleapis.com');

/* Main Body Font */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #090a0f;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(45deg, #3b82f6, #00f5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    animation: fadeIn 1.2s ease-in-out;
}

.subtitle {
    text-align: center;
    color: #8e8e93;
    font-size: 15px;
    margin-bottom: 30px;
}

/* PREMIUM GLASS CARD CONTAINERS */
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 24px;
    margin-bottom: 22px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
}

/* TELEMETRY LABELS */
.matrix-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.06);
    padding: 6px 14px;
    border-radius: 10px;
    margin-right: 6px;
    margin-bottom: 8px;
    color: #00f5ff;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid rgba(255, 255, 255, 0.02);
}

.scenario-surge {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.scenario-recession {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* ANIMATIONS */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("<div class='title'>📊 Sales Predictive Studio Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deterministic Time-Series Forecasting Architecture & Multi-Horizon Telemetry</div>", unsafe_allow_html=True)

# ==========================================
# INPUT LAYER MODULE (GLASS CARD UI)
# ==========================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### ⚙️ Time-Series Operational Matrix Inputs")

col1, col2 = st.columns(2)
with col1:
    historical_baseline = st.number_input("📈 Previous Week Volumetric Close ($):", min_value=100, max_value=500000, value=14500, step=500)
    rolling_mean_7d = st.number_input("🔄 7-Day Rolling Historical Average ($):", min_value=100, max_value=500000, value=13800, step=500)
    seasonal_lag_30d = st.number_input("📅 30-Day Cyclical Month-Lag Volatility ($):", min_value=100, max_value=500000, value=15200, step=500)

with col2:
    selected_day = st.selectbox("📆 Target Projection Start Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    marketing_spend = st.slider("🚀 Marketing Campaign Spend Vector ($):", 0, 15000, 2500, step=250)
    
    # Premium Feature Integration: Scenario Simulator Injector Dropdown
    macro_scenario = st.selectbox("🌐 Simulated Market Macro Environment", ["Standard Baseline Optimization", "Q4 Peak Holiday Surge", "Market Contraction Recession"])

forecast_btn = st.button("⚡ Run Deterministic Horizon Forecasting Pipeline", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# DETREMINISTIC FORECASTING ENGINE CONSTRUCTS
# ==========================================
def calculate_horizon_predictions(hist, roll, lag, start_day, ads, scenario):
    day_indices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}
    
    # Scenario Scalar Modifiers Layer
    scenario_scalars = {"Standard Baseline Optimization": 1.00, "Q4 Peak Holiday Surge": 1.38, "Market Contraction Recession": 0.74}
    current_scalar = scenario_scalars[scenario]
    
    # Mathematical linear feature matrix vector calculations
    feature_mix_base = (hist * 0.30) + (roll * 0.45) + (lag * 0.25)
    marketing_coefficient_boost = float(ads) * 0.92
    
    # Generate full 7-day predictive curves rolling from chosen start index day smoothly
    start_idx = day_indices.index(start_day)
    ordered_days = day_indices[start_idx:] + day_indices[:start_idx]
    
    future_horizon_curve = []
    decay_accumulation_factor = 1.0
    
    for i, day in enumerate(ordered_days):
        # Introduce a slight decay factor across a 7-day timeline window to simulate rolling tracking horizons
        decay_accumulation_factor -= (i * 0.015)
        day_prediction = (feature_mix_base + marketing_coefficient_boost) * day_weights[day] * current_scalar * decay_accumulation_factor
        future_horizon_curve.append((day, float(day_prediction)))
        
    return future_horizon_curve

# ==========================================
# MAIN EXECUTION COMPILATION ROUTING
# ==========================================
if forecast_btn:
    if historical_baseline <= 0 or rolling_mean_7d <= 0:
        st.error("🚨 Mathematical Error: Input metric sequences must exceed zero bounds.")
    else:
        with st.spinner("🧠 Initializing Statistical Regression Pipelines..."):
            time.sleep(1.0)
            
        # Run local algebraic matrix calculation curves
        forecast_timeline_curve = calculate_horizon_predictions(
            historical_baseline, rolling_mean_7d, seasonal_lag_30d,
            selected_day, marketing_spend, macro_scenario
        )
        
        target_day_name, predicted_revenue = forecast_timeline_curve[0]
        
        # Display contextual scenario tags onto execution workspace
        st.markdown("### 🍿 Analytical Horizon Verdicts")
        if macro_scenario == "Q4 Peak Holiday Surge":
            st.markdown("<span class='matrix-tag scenario-surge'>⚡ Macro Trigger: Active Volumetric Demand Surge (1.38x)</span>", unsafe_allow_html=True)
        elif macro_scenario == "Market Contraction Recession":
            st.markdown("<span class='matrix-tag scenario-recession'>⚠️ Macro Trigger: Active System Volume Deflation (0.74x)</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='matrix-tag'>🟢 Macro Trigger: Standard Operation Vectors</span>", unsafe_allow_html=True)
            
        # TELEMETRY DISPLAY GAUGES
        col_m1, col_v2 = st.columns(2)
        with col_m1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.metric(
                label=f"🎯 Target Revenue Forecast ({target_day_name})", 
                value=f"${predicted_revenue:,.2f}",
                delta=f"{(predicted_revenue - rolling_mean_7d) / rolling_mean_7d * 100:.1f}% Variance vs 7D Mean"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_v2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            demand_status = "📈 EXPANSION WAVE" if predicted_revenue > rolling_mean_7d else "📉 COMPRESSION WAVE"
            st.metric(label="🔄 Projected Operational Vector", value=demand_status)
            st.markdown("</div>", unsafe_allow_html=True)

        # RENDER ADVANCED PROGRESSIVE TIMELINE LINE GRAPH
        st.markdown("### 📈 7-Day Progressive Target Predictive Horizon Curve")
        
        days_axis = [item[0] for item in forecast_timeline_curve]
        revenue_axis = [item[1] for item in forecast_timeline_curve]
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=days_axis, y=revenue_axis,
            mode='lines+markers+text',
            name='AI Forecast Curve',
            text=[f"${val/1000:.1f}k" for val in revenue_axis],
            textposition="top center",
            line=dict(color='#00f5ff', width=3),
            marker=dict(size=8, color='#3b82f6', line=dict(color='#fff', width=1.5))
        ))
        
        fig_curve.update_layout(
            title=f"7-Day Rolling Revenue Velocity Plan Path (Injected Scenario: {macro_scenario})",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#fff",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Projected Currency Metrics ($)"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=340
        )
        st.plotly_chart(fig_curve, width='stretch')

        # AUDIT APPEND LOOP
        st.session_state.history.append({
            "Timestamp": time.strftime("%H:%M:%S"),
            "Start Day": selected_day,
            "Macro Environment": macro_scenario,
            "Marketing Input": f"${marketing_spend:,}",
            "Target Projection Yield": f"${predicted_revenue:,.2f}"
        })

# =========================
# SYSTEM HISTORY DATA LOGGER
# =========================
st.markdown("---")
st.markdown("### 🎞️ Enterprise Session Audit History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, width='stretch')
    
    if st.button("🗑️ Reset Session Logs"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Studio session audit logs are currently empty. Run an evaluation pipeline to track telemetry data 🎬")

# =========================
# FOOTER
# =========================
st.markdown("---")
