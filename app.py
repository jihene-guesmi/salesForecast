import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ==========================================
# DASHBOARD CONFIGURATION (LIGHT THEME)
# ==========================================
st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Custom Light & Clean CSS Styling
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Base application styling */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}

/* Clear, light panel cards */
.light-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

/* Metric text formatting */
.metric-title {
    font-size: 13px;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# BACKEND PRODUCT INVENTORY & SALES DATA
# ==========================================
# This acts as your local backend database
product_data = pd.DataFrame({
    "Product Name": ["Organic Fertilizers", "Premium Crop Seeds", "Drip Irrigation Kits", "Eco Pest Controls", "Soil Testing Probes"],
    "Stock Status": ["Available", "Available", "Low Stock", "Available", "Out of Stock"],
    "Units Available": [450, 1200, 15, 310, 0],
    "Units Sold This Month": [120, 840, 95, 210, 45],
    "Base Price ($)": [45.00, 15.00, 250.00, 30.00, 85.00]
})

# Sort to isolate the most sold product dynamically
most_sold_product = product_data.sort_values(by="Units Sold This Month", ascending=False).iloc[0]

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("<h1 style='color: #0f172a; font-weight: 800; margin-bottom: 0;'>🛍️ Retail Sales & AI Prediction Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 15px; margin-bottom: 25px;'>Monitor live inventory availability, track top-selling products, and run predictive sales modeling templates.</p>", unsafe_allow_html=True)

# Top Telemetry Highlight KPI Cards Rows
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.markdown(f"""
    <div class='light-card'>
        <div class='metric-title'>🔥 Current Top Selling Product</div>
        <div class='metric-value' style='color: #2563eb;'>{most_sold_product["Product Name"]}</div>
        <div style='font-size: 12px; color: #64748b; margin-top: 4px;'>{most_sold_product["Units Sold This Month"]} units moved this month</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    available_count = len(product_data[product_data["Stock Status"] != "Out of Stock"])
    st.markdown(f"""
    <div class='light-card'>
        <div class='metric-title'>📦 Active Product Catalog</div>
        <div class='metric-value'>{available_count} / {len(product_data)} Items Available</div>
        <div style='font-size: 12px; color: #64748b; margin-top: 4px;'>1 item currently out of stock</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    total_inventory_revenue = (product_data["Units Sold This Month"] * product_data["Base Price ($)"]).sum()
    st.markdown(f"""
    <div class='light-card'>
        <div class='metric-title'>📈 Monthly Gross Revenue</div>
        <div class='metric-value' style='color: #16a34a;'>${total_inventory_revenue:,.2f}</div>
        <div style='font-size: 12px; color: #64748b; margin-top: 4px;'>Calculated from direct product sales</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN INTERFACE LAYOUT GRID SPREAD
# ==========================================
col_left_inputs, col_right_charts = st.columns([1, 1.8])

# LEFT COLUMN: FEATURE INPUT SELECTORS (MATCHING YOUR NOTEBOOK MODELS)
with col_left_inputs:
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color: #0f172a;'>⚙️ Configure Model Prediction Features</h3>", unsafe_allow_html=True)
    st.write("Select the necessary features used to train your regression notebooks to simulate future revenue labels:")
    
    # 1. Select Target Product Group Feature
    selected_prod = st.selectbox("📦 Target Product Focus:", product_data["Product Name"])
    prod_row = product_data[product_data["Product Name"] == selected_prod].iloc[0]
    
    # 2. Select Temporal Day Feature
    target_day = st.selectbox("📆 Forecast Target Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    
    # 3. Select Operational Strategy Features
    marketing_budget = st.slider("🚀 Advertising Campaign Spend ($):", 0, 15000, 2500, step=500)
    is_holiday = st.checkbox("🎉 Target Horizon is a National Holiday")
    competitor_discount = st.checkbox("🚨 Competitors Running Price Cuts / Discounts")
    
    forecast_btn = st.button("⚡ Run Sales Model Inference", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# BACKEND ALGEBRAIC PREDICTION MODEL SCALARS
# ==========================================
# Day seasonality parameters matching model configurations
day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}

# Compute baseline calculation using product parameters and marketing boost features
base_revenue_mix = (prod_row["Base Price ($)"] * prod_row["Units Sold This Month"] * 0.35) + (float(marketing_budget) * 0.88)
calculated_prediction_label = base_revenue_mix * day_weights[target_day]

# Apply environmental scenario multipliers
if is_holiday:
    calculated_prediction_label *= 1.42
if competitor_discount:
    calculated_prediction_label *= 0.82

# RIGHT COLUMN: VISUAL DASHBOARD CHARTS (LIGHT & CLEAN COLORS)
with col_right_charts:
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0; color: #0f172a;'>🎯 Predicted AI Sales Label for {selected_prod}</h3>", unsafe_allow_html=True)
    
    # Large readable prediction metric block
    st.metric(
        label=f"Expected Single-Day Revenue Velocity on {target_day}", 
        value=f"${calculated_prediction_label:,.2f}"
    )
    
    # Strategic Action Directives Engine
    st.markdown("---")
    st.markdown("📋 **Strategic Advisory Action Plan:**")
    if is_holiday and not competitor_discount:
        st.success(f"Demand spikes predicted due to holiday filters! Stock up on **{selected_prod}** immediately to capitalize on high transactional volume waves.")
    elif competitor_discount:
        st.warning(f"Competitor pricing discount waves flagged. Guard sales volume metrics for **{selected_prod}** by highlighting value bundles or introducing loyalty rewards instead of cutting margins.")
    else:
        st.info("Market conditions remain stable. Maintain lean inventory replacement protocols and keep standard campaign pacing loops steady.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 7-DAY sales VISUALIZATION TREND LINE
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color: #0f172a;'>📈 7-Day Rolling Revenue Progression Horizon</h4>", unsafe_allow_html=True)
    
    week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    curve_data = []
    for day in week_days:
        day_pred = base_revenue_mix * day_weights[day]
        if is_holiday:
            day_pred *= 1.42
        if competitor_shock:
            day_pred *= 0.82
        curve_data.append((day, round(day_pred, 2)))
        
    chart_df = pd.DataFrame(curve_data, columns=["Day of Week", "Predicted Revenue ($)"])
    
    # Generate clean, light-colored graph layout parameters
    fig_line = px.line(
        chart_df, 
        x="Day of Week", 
        y="Predicted Revenue ($)", 
        markers=True,
        color_discrete_sequence=["#2563eb"]  # Clean corporate royal blue line
    )
    
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=11, color="#475569"),
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0', title=None),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# LOWER LAYER: LIVE STOCK AVAILABILITY MATRIX TABLE
st.markdown("<div class='light-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color: #0f172a;'>📊 Real-Time Backend Inventory Matrix</h3>", unsafe_allow_html=True)
st.write("Current snapshot values loaded from your active database tracking system script:")

# Style data frame formatting for easy business overview scanning
st.dataframe(
    product_data.style.format({"Base Price ($)": "${:.2f}"}),
    use_container_width=True,
    hide_index=True
)
st.markdown("</div>", unsafe_allow_html=True)

# FOOTER CLOSING RECONCILIATIONS
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 11px;'>🛍️ User-Optimized Light Theme Enterprise Sales Dashboard Portfolio Ecosystem Deployed</div>", unsafe_allow_html=True)
