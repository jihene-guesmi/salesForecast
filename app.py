import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# ENTERPRISE LEVEL LEVEL CORE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Nexus Predictive Sales Engine Pro",
    page_icon="🔮",
    layout="wide"  # Maximizes grid real-estate scanning capabilities
)

# Initialize deep running state tracking arrays if absent from the container runtime
if "telemetry_vault" not in st.session_state:
    st.session_state.telemetry_vault = [
        {"Timestamp": "14:22:05", "Selected Day": "Friday", "Macro Vector": "Peak Holiday Surge", "Ad Spend ($)": 5000, "Calculated Label ($)": 28435.50, "Status": "🔥 Peak Operations"},
        {"Timestamp": "15:40:12", "Selected Day": "Monday", "Macro Vector": "Normal Baseline", "Ad Spend ($)": 1500, "Calculated Label ($)": 13920.10, "Status": "🟢 Stable Optimal"},
    ]

# Premium High-Density Glassmorphism CSS Matrix Themes
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Base System Application Resets */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #05070c;
    color: #f9fafb;
}

/* Global Glassmorphic Container Core Blocks */
.dashboard-block {
    background: linear-gradient(135deg, rgba(13, 17, 23, 0.7) 0%, rgba(10, 12, 18, 0.8) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    margin-bottom: 22px;
}

/* Executive KPI Analytics Deck Layouts */
.kpi-deck-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    text-align: left;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.kpi-card-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #9ca3af;
    font-weight: 600;
    letter-spacing: 0.08em;
}

.kpi-card-value {
    font-size: 34px;
    font-weight: 900;
    margin-top: 8px;
    margin-bottom: 4px;
}

/* High-Contrast Neon Visual Lighting Classes */
.neon-cyan { color: #00f5ff; text-shadow: 0 0 15px rgba(0,245,255,0.3); }
.neon-emerald { color: #10b981; text-shadow: 0 0 15px rgba(16,185,129,0.3); }
.neon-amber { color: #f59e0b; text-shadow: 0 0 15px rgba(245,158,11,0.3); }

/* Strategy Directive Blocks */
.directive-container {
    padding: 16px;
    border-radius: 12px;
    margin-top: 12px;
    font-size: 13px;
    line-height: 1.6;
}
.directive-market { background: rgba(59, 130, 246, 0.06); border-left: 4px solid #3b82f6; }
.directive-supply { background: rgba(16, 185, 129, 0.06); border-left: 4px solid #10b981; }
.directive-risk { background: rgba(239, 68, 68, 0.06); border-left: 4px solid #ef4444; }

/* Status Chip Badges */
.pill-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}
.pill-blue { background: rgba(59, 130, 246, 0.12); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.2); }
.pill-green { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
.pill-red { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ALGEBRAIC MODEL MATRIX ENGINE
# ==============================================================================
def execute_tensor_forecast_pipeline(hist_base, roll_avg, seasonal_lag, day, ads, macro, shock):
    day_keys = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_coefficients = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}
    macro_coefficients = {"Normal Baseline": 1.00, "Q4 Peak Holiday Surge": 1.42, "Market Contraction Recession": 0.71}
    
    # 🧠 EXPLICIT WEIGHTED FEATURE ATTRIBUTION MATRIX
    w1, w2, w3 = 0.25, 0.45, 0.30
    aggregated_feature_mix = (float(hist_base) * w1) + (float(roll_avg) * w2) + (float(seasonal_lag) * w3)
    
    marketing_return_boost = float(ads) * 0.88
    shock_multiplier = 0.82 if shock else 1.00
    global_macro_scalar = macro_coefficients[macro]
    
    start_idx = day_keys.index(day)
    chronological_sequence = day_keys[start_idx:] + day_keys[:start_idx]
    
    compiled_projection_yield = []
    horizon_decay = 1.0
    
    for i, step_day in enumerate(chronological_sequence):
        # Rolling tracking horizon structural decay simulation vector parameters
        horizon_decay -= (i * 0.012)
        step_label_yield = (aggregated_feature_mix + marketing_return_boost) * day_coefficients[step_day] * global_macro_scalar * shock_multiplier * horizon_decay
        compiled_projection_yield.append((step_day, float(step_label_yield)))
        
    return compiled_projection_yield, {
        "W1_LastWeek": w1, "W2_Rolling7D": w2, "W3_MonthLag": w3,
        "Active_Day_Coefficient": day_coefficients[day],
        "Active_Macro_Scalar": global_macro_scalar,
        "Active_Price_Shock": shock_multiplier
    }

# ==============================================================================
# MAIN SYSTEM WORKSPACE HEADERS & INTERACTIVE TABS
# ==============================================================================
st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# System Command Navigation Ribbon
tab_core_studio, tab_vector_dictionary = st.tabs([
    "📊 Enterprise Forecasting Control Center", 
    "📚 Model Feature & Weight Architecture"
])

# ==============================================================================
# TAB VALUE 1: EXECUTIVE ANALYSIS ENGINE
# ==============================================================================
with tab_core_studio:
    # 🕹️ SIDEBAR INDEPENDENT MODEL FEATURES PANEL
    st.sidebar.markdown("### 💾 Independent Feature Config Deck")
    
    st.sidebar.markdown("**Temporal & Environmental Features:**")
    target_start_day = st.sidebar.selectbox("📆 Selected Horizon Start Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    macro_environment_feature = st.sidebar.selectbox("🌐 Simulated Market Macro Environment", ["Normal Baseline", "Q4 Peak Holiday Surge", "Market Contraction Recession"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Strategic Control Features:**")
    advertising_budget_feature = st.sidebar.slider("🚀 Allocated Campaign Ads Budget ($)", 0, 20000, 4500, step=500)
    competitor_shock_feature = st.sidebar.checkbox("🚨 Apply Competitor Aggressive Price Cuts / Shock")

    # Main Grid Shell: Left Input Vector Array Block, Right Execution Workspace Canvas
    left_feature_deck, right_analytics_canvas = st.columns([1.1, 2.4])

    with left_feature_deck:
        st.markdown("<div class='dashboard-block'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-weight:700;'>📝 Historical Baseline Features</h3>", unsafe_allow_html=True)
        
        # Capture raw continuous business input metrics
        last_week_sales_feat = st.number_input("💵 Last Week Total Close Sales ($):", min_value=100, max_value=2000000, value=14500, step=500)
        rolling_7d_mean_feat = st.number_input("🔄 7-Day Rolling Operational Mean ($):", min_value=100, max_value=2000000, value=13800, step=500)
        month_seasonal_lag_feat = st.number_input("📅 30-Day Cyclical Month Lag ($):", min_value=100, max_value=2000000, value=15200, step=500)
        
        trigger_pipeline_loop = st.button("⚡ Execute Model Inference Loop", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Compile model arrays instantly to capture active runtime parameters
    model_output_curve, network_attribution_weights = execute_tensor_forecast_pipeline(
        last_week_sales_feat, rolling_7d_mean_feat, month_seasonal_lag_feat,
        target_start_day, advertising_budget_feature, macro_environment_feature, competitor_shock_feature
    )
    
    # Isolate calculated model labels
    target_day_name_label = model_output_curve[0][0]
    target_day_yield_label = float(model_output_curve[0][1])
    baseline_comparison_value = float(rolling_7d_mean_feat)
    cumulative_7d_horizon_label = sum([val for _, val in model_output_curve])

    # RIGHT COLUMN EXECUTION WORKSPACE
    with right_analytics_canvas:
        # 1. THREE-COLUMN METRIC GAUGES DECK
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            variance_pct = ((target_day_yield_label - baseline_comparison_value) / baseline_comparison_value) * 100
            neon_metric_class = "neon-emerald" if variance_pct >= 0 else "neon-amber"
            st.markdown(f"""
            <div class='kpi-deck-card'>
                <div class='kpi-card-title'>Target Day Predicted Label ({target_day_name_label})</div>
                <div class='kpi-card-value {neon_metric_class}'>${target_day_yield_label:,.2f}</div>
                <div style='font-size:11px; font-weight:600; color:{"#10b981" if variance_pct >= 0 else "#ef4444"}'>
                    {"▲" if variance_pct >= 0 else "▼"} {abs(variance_pct):.1f}% Variance vs 7D Avg
                </div>
            </div>
            """, unsafe_allow_html=True)
            
       with kpi_col2:
            st.markdown(f"""
            <div class='kpi-deck-card'>
                <div class='kpi-card-title'>Cumulative 7-Day Horizon Label</div>
                <div class='kpi-value neon-cyan'>${cumulative_7d_horizon_label:,.2f}</div>
                <div style='font-size:11px; color:#9ca3af;'>Aggregate Expected Cashflow Volume</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            # Active status badge logic maps explicitly to risk/opportunity feature flags
            if macro_environment_feature == "Q4 Peak Holiday Surge" and not competitor_shock_feature:
                badge_markup = "<span class='pill-badge pill-green'>🔥 Peak Market Expansion</span>"
            elif macro_environment_feature == "Market Contraction Recession" or competitor_shock_feature:
                badge_markup = "<span class='pill-badge pill-red'>⚠️ Volatility Risk Matrix</span>"
            else:
                badge_markup = "<span class='pill-badge pill-blue'>🟢 Standard Optimization</span>"
                
            st.markdown(f"""
            <div class='kpi-deck-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: flex-start;'>
                <div class='kpi-card-title' style='margin-bottom:8px;'>Model Operational Status</div>
                {badge_markup}
            </div>
            """, unsafe_allow_html=True)

        # 2. EXECUTIVE CORE VISUAL TREND GRAPH
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        timeline_dataframe = pd.DataFrame(model_output_curve, columns=["Target Horizon Day", "Predicted Revenue Yield ($)"])
        
        # Dual-Axis/High-Contrast Plotly line engine tracks values across the 7D sequence
        fig_time_series = px.line(
            timeline_dataframe, 
            x="Target Horizon Day", 
            y="Predicted Revenue Yield ($)",
            title="🎯 Multi-Horizon Time-Series Predictive Revenue Velocity Curve",
            markers=True,
            text=timeline_dataframe["Predicted Revenue Yield ($)"].apply(lambda x: f"${x/1000:.1f}k")
        )
        
        fig_time_series.update_traces(
            line=dict(color='#00f5ff', width=3.5),
            marker=dict(size=8, color='#007aff', line=dict(color='#fff', width=1)),
            textposition="top center"
        )
        
        fig_time_series.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color="#8e8e93"),
            title_font=dict(size=14, color="#ffffff", family="Inter"),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff')),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#fff'), title=None),
            margin=dict(l=10, r=10, t=50, b=10),
            height=280
        )
        st.plotly_chart(fig_time_series, use_container_width=True)

    # ==============================================================================
    # STRATEGIC EXECUTIVE DECISION DIRECTIVE ENGINE
    # ==============================================================================
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    lower_grid_col1, lower_grid_col2 = st.columns(2)

    with lower_grid_col1:
        st.markdown("<div class='dashboard-block' style='height:100%;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-weight:700;'>🧠 Automated Strategic Intelligence Agent</h3>", unsafe_allow_html=True)
        st.write("The agent processed your uploaded features and compiled these prescriptive corporate recommendations:")
        
        # Dynamic Marketing Allocation Directive Calculation
        if advertising_budget_feature >= 8000 and not competitor_shock_feature:
            st.markdown("""
            <div class='directive-container directive-market'>
                <strong>🚀 Premium Marketing Spend Allocation Recommendation:</strong><br>
                Your high ad spend is highly optimized for the current environment. Focus campaign targets on digital channels and premium keywords to lock down consumer acquisition loops during this high-velocity volume wave.
            </div>
            """, unsafe_allow_html=True)
        elif competitor_shock_feature:
            st.markdown("""
            <div class='directive-container directive-risk'>
                <strong>🚨 Pricing War Combat Counter-Strategy:</strong><br>
                Active competitor discount shocks are threatening margin profiles. Avoid competing on price cuts alone; instead, launch bundle packs, value-adds, and point-based loyalty programs to defend market volume share without killing margins.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='directive-container directive-market'>
                <strong>📈 Balanced Customer Acquisition Budgeting Strategy:</strong><br>
                Ad spend is running at conservative baseline rates. Consider increasing the campaign budget by $2,500 to expand outreach and drive more volume into the sales pipeline.
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Supply Chain / Inventory Regulations Calculations
        if target_day_yield_label > baseline_comparison_value * 1.15:
            st.markdown("""
            <div class='directive-container directive-supply'>
                <strong>📦 Mandatory Inventory Holding Expansion Order:</strong><br>
                The predictive curve identifies a revenue surge wave. Increase safety stock thresholds by 20% across main regional fulfillment centers over the next 48 hours to avoid stockouts and protect fulfillments.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='directive-container directive-supply'>
                <strong>🔄 Lean Distribution Replenishment Regulation:</strong><br>
                Future demand curves match historical baselines cleanly. Maintain lean inventory configurations to reduce holding costs and protect capital allocations.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Immutable Session Audit Database Logs Tracker System Panel
    with lower_grid_col2:
        st.markdown("<div class='dashboard-block' style='height:100%;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-weight:700;'>🎞️ Enterprise Session Telemetry Audit Log</h3>", unsafe_allow_html=True)
        st.write("Running list of all active model simulation scenarios compiled during this working container session:")
        
        if trigger_pipeline_loop:
            st.session_state.telemetry_vault.append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "Selected Day": target_start_day,
                "Macro Vector": macro_environment_feature if not competitor_shock_feature else "🚨 Price Shock Wars",
                "Ad Spend ($)": int(advertising_budget_feature),
                "Calculated Label ($)": round(target_day_yield_label, 2),
                "Status": "🔥 Peak Operations" if target_day_yield_label > last_week_sales_feat * 1.1 else "🟢 Growth Wave" if target_day_yield_label >= last_week_sales_feat else "🔴 Volume Compression"
            })
            
        dataframe_audit_vault = pd.DataFrame(st.session_state.telemetry_vault[::-1]) 
        st.dataframe(dataframe_audit_vault, use_container_width=True, height=145)
        
        if st.button("🗑️ Reset Vault Telemetry Databases", use_container_width=True):
            st.session_state.telemetry_vault = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB VALUE 2: TRANSPARENT MATHEMATICAL MODEL VECTOR CONFIGURATION DEFINITIONS
# ==============================================================================
with tab_vector_dictionary:
    st.markdown("<div class='dashboard-block'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-weight:700;'>📚 Neural Layer Vector & Attribution Dictionary</h3>", unsafe_allow_html=True)
    st.write("This transparency tab shows exactly how your independent variables (Features) are calculated into target business answers (Labels) inside the model layer:")
    
    col_dict1, col_dict2 = st.columns(2)
    with col_dict1:
        st.markdown(f"""
        **📂 Captured Independent Feature Variables:**
        - **Last Week Sales Feature:** `${last_week_sales_feat:,}` (Acts as a structural anchor)
        - **Rolling 7-Day Average Feature:** `${rolling_7d_mean_feat:,}` (Captures current velocity)
        - **Month Seasonal Lag Feature:** `${month_seasonal_lag_feat:,}` (Captures long-term cycles)
        - **Temporal Start Day Feature:** `{target_start_day}` (Applies weekly multipliers)
        - **Marketing Budget Feature:** `${advertising_budget_feature:,}` (Adds linear attribution boosts)
        - **Macro Environment Feature:** `{macro_environment_feature}` (Applies market scaling filters)
        - **Competitor Shock Flag Feature:** `{competitor_shock_feature}` (Models pricing war margin reductions)
        """)
        
    with col_dict2:
        st.markdown(f"""
        **⚙️ Current Mathematical Attribution Weights:**
        - **Last Week Attribution Weight (W1):** `{network_attribution_weights['W1_LastWeek']:.2f}`
        - **Rolling 7-Day Mean Weight (W2):** `{network_attribution_weights['W2_Rolling7D']:.2f}`
        - **Month Seasonal Lag Weight (W3):** `{network_attribution_weights['W3_MonthLag']:.2f}`
- Active Day Multiplier Weight: {network_attribution_weights['Active_Day_Coefficient']:.2f}x
- Active Macro Scalar Weight Filter: {network_attribution_weights['Active_Macro_Scalar']:.2f}x
- Active Pricing War Suppression Multiplier: {network_attribution_weights['Active_Price_Shock']:.2f}x
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SYSTEM PANEL RECONCILIATION FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 11px;'>🍿 Enterprise Decision Infrastructure Engine Ecosystem | Production-Grade Multi-Layer Vector Ensembles Online</div>", unsafe_allow_html=True)
