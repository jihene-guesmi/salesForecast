import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Professional Sales Dashboard API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "sales_forecast_model.pkl")
model = joblib.load(model_path)

# Load sales data
csv_path = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")
data_sales = pd.read_csv(csv_path)

# Encode products
product_le = LabelEncoder()
data_sales['product_encoded'] = product_le.fit_transform(data_sales['product_name'])

# Encode aisle if exists
if 'aisle' in data_sales.columns:
    aisle_le = LabelEncoder()
    data_sales['aisle_encoded'] = aisle_le.fit_transform(data_sales['aisle'])
else:
    data_sales['aisle_encoded'] = 0

# Unique products for dropdown
product_info = data_sales[['product_name','product_encoded','aisle_encoded']].drop_duplicates()

@app.get("/products")
def get_products():
    return product_info.to_dict(orient="records")

@app.get("/top_products")
def get_top_products():
    top = data_sales.groupby("product_name")["quantity"].sum().sort_values(ascending=False).head(10)
    return [{"product_name": p, "units_sold": int(u)} for p, u in top.items()]

@app.get("/weekly_sales/{product_name}")
def get_weekly_sales(product_name: str):
    df = data_sales[data_sales['product_name']==product_name].sort_values('transaction_date')
    df['week'] = pd.to_datetime(df['transaction_date']).dt.isocalendar().week
    weekly = df.groupby('week')['quantity'].sum().reset_index()
    return {"weeks": weekly['week'].tolist(), "units": weekly['quantity'].tolist()}

@app.get("/promotion_ratio/{product_name}")
def get_promotion_ratio(product_name: str):
    df = data_sales[data_sales['product_name']==product_name]
    promo = df['promotion'].sum() if 'promotion' in df.columns else 0
    non_promo = len(df) - promo
    return {"promotion": int(promo), "normal": int(non_promo)}

@app.get("/top_trending")
def get_top_trending():
    df = data_sales.copy()
    df['week'] = pd.to_datetime(df['transaction_date']).dt.isocalendar().week
    last_week = df['week'].max()
    prev_week = last_week - 1
    last = df[df['week']==last_week].groupby('product_name')['quantity'].sum()
    prev = df[df['week']==prev_week].groupby('product_name')['quantity'].sum()
    trend = ((last - prev)/prev.replace(0,1)*100).sort_values(ascending=False).head(10)
    return [{"product_name": p, "trend_percent": round(t,2)} for p,t in trend.items()]

class ProductInput(BaseModel):
    product_name: str
    unit_price: float = Field(..., gt=0)
    promotion: int = Field(..., ge=0, le=1)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    quarter: int = Field(..., ge=1, le=4)
    lag_1: float
    lag_7: float
    rolling_7: float
    product_encoded: int
    aisle_encoded: int
    is_holiday: int = 0

@app.post("/predict")
def predict_sales(product: ProductInput):
    # Only use features model was trained on
    features = ["unit_price","promotion","day_of_week","month","quarter",
                "lag_1","lag_7","rolling_7","product_encoded","aisle_encoded"]
    df = pd.DataFrame([{col: getattr(product, col) for col in features}])
    
    prediction = model.predict(df)[0]

    # Trend logic
    trend = "stable"
    if product.lag_1 - product.lag_7 > 0.5: trend="increasing"
    elif product.lag_1 - product.lag_7 < -0.5: trend="decreasing"

    recommendation = []
    if product.promotion==1: recommendation.append("Promotion active → sales expected to increase")
    if product.is_holiday==1: recommendation.append("Holiday → higher sales expected")
    if product.day_of_week>=5: recommendation.append("Weekend → higher sales expected")
    if not recommendation: recommendation.append("No special effect → stable sales")

    interpretation = f"Trend '{trend}' because recent sales (Lag1={product.lag_1} vs Lag7={product.lag_7})"
    price_effect = f"Discounted price: {product.unit_price*0.9:.2f}" if product.promotion==1 else f"Price: {product.unit_price:.2f}"
    historical_sales = [product.lag_1, product.lag_7, product.rolling_7]

    return {
        "product": product.product_name,
        "predicted_units_sold": round(float(prediction),2),
        "trend": trend,
        "historical_sales": historical_sales,
        "recommendation": recommendation,
        "interpretation": interpretation,
        "price_effect": price_effect
    }
