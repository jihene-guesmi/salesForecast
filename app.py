import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ==========================================
# SIMPLE DASHBOARD CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Simple Sales Prediction Dashboard")
st.write("Adjust your business features to instantly calculate next week's expected sales and view strategic recommendations.")

st.markdown("---")

# ==========================================
# ⚙️ INPUT BUSINESS FEATURES
# ==========================================
st.subheader("🛠️ Step 1: Configure Your Business Features")

col1, col2 = st.columns(2)

with col1:
    historical_baseline = st.number_input("💵 Average Weekly Sales ($):", min_value=100, max_value=1000000, value=15000, step=500)
    target_day = st.selectbox("📆 Target Day to Predict:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    marketing_budget = st.slider("🚀 Marketing Campaign Spend ($):", 0, 20000, 3000, step=500)

with col2:
    is_holiday = st.checkbox("🎉 Is the Target Day a Holiday? (+42% Surge)")
    has_competitor_discount = st.checkbox("🚨 Are Competitors Running Discounts? (-18% Drop)")

st.markdown("---")

# ==========================================
# 🧠 ALGEBRAIC MODEL CALCULATION
# ==========================================
day_weights = {"Monday": 0.92, "Tuesday": 0.96, "Wednesday": 1.02, "Thursday": 1.05, "Friday": 1.22, "Saturday": 1.28, "Sunday": 0.85}

# Calculate baseline with marketing budget boost
base_prediction = float(historical_baseline) + (float(marketing_budget) * 0.85)

# Apply day seasonality modifier
calculated_sales_label = base_prediction * day_weights[target_day]

# Apply holiday or competitor event multipliers
if is_holiday:
    calculated_sales_label *= 1.42
if has_competitor_discount:
    calculated_sales_label *= 0.82

# ==========================================
# 🎯 DYNAMIC STRATEGIC RECOMMENDATIONS
# ==========================================
st.subheader("🎯 Step 2: AI Predicted Sales Label & Action Plan")

# Main KPI Scorecard Display
st.metric(
    label=f"Expected Sales Label for {target_day}", 
    value=f"${calculated_sales_label:,.2f}"
)

# Plain Business Actions Based Directly on Active Features
if is_holiday and not has_competitor_discount:
    st.success("🔥 **Marketing & Supply Plan:** Demand will be high due to the holiday! Increase your stock levels immediately and maximize ad spend to capture maximum shoppers.")
elif has_competitor_discount:
    st.warning("🚨 **Risk Counter-Strategy:** Competitors are cutting prices. Avoid entering a margin-killing price war. Instead, offer value bundles or run loyalty point promotions to keep your customers.")
else:
    st.info("🟢 **Standard Operations:** Market forces are steady. Maintain normal item replenishment schedules and keep your baseline marketing spend stable.")

st.markdown("---")

# ==========================================
# 📈 SIMPLE 7-DAY VISUALIZATION
# ==========================================
st.subheader("📈 Step 3: 7-Day Expected Sales Progression")

future_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
seven_day_yields = []

for day in future_days:
    day_yield = base_prediction * day_weights[day]
    if is_holiday:
        day_yield *= 1.42
    if has_competitor_discount:
        day_yield *= 0.82
    seven_day_yields.append((day, round(day_yield, 2)))

chart_df = pd.DataFrame(seven_day_yields, columns=["Day of Week", "Predicted Sales ($)"])

# Clean, simple line chart showing the progression loop
fig = px.line(
    chart_df, 
    x="Day of Week", 
    y="Predicted Sales ($)",
    markers=True,
    title="Expected Weekly Revenue Progression Curve"
)

fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)
