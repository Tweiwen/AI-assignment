import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide"
)

# Header Section
st.title("🎬 Netflix Customer Churn Prediction System")
st.markdown("Enter customer details below to predict account cancellation risk using Machine Learning (Random Forest).")
st.divider()

# Cache model training so it runs instantly
@st.cache_resource
def train_model():
    df = pd.read_csv('netflix_customer_churn.csv')
    X = df.drop(columns=['customer_id', 'churned'])
    y = df['churned']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Train / load model
with st.spinner("Initializing Model..."):
    pipeline = train_model()

# ---------------------------------------------------------
# MAIN HOMEPAGE INPUT FORM (GRID LAYOUT)
# ---------------------------------------------------------
st.subheader("📋 Customer Details Input")

# Row 1: Demographics & Account Setup
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col3:
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
with col4:
    monthly_fee = st.number_input("Monthly Fee ($)", min_value=5.0, max_value=30.0, value=13.99)

# Row 2: Usage Behavior
col5, col6, col7, col8 = st.columns(4)

with col5:
    watch_hours = st.number_input("Total Watch Hours", min_value=0.0, max_value=500.0, value=25.0)
with col6:
    avg_watch_time_per_day = st.number_input("Avg Daily Watch Time (Hrs)", min_value=0.0, max_value=24.0, value=1.5)
with col7:
    last_login_days = st.slider("Days Since Last Login", min_value=0, max_value=60, value=10)
with col8:
    number_of_profiles = st.slider("Number of Profiles", min_value=1, max_value=5, value=2)

# Row 3: Preferences & Location
col9, col10, col11, col12 = st.columns(4)

with col9:
    region = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
with col10:
    device = st.selectbox("Device Used", ["TV", "Mobile", "Desktop", "Tablet"])
with col11:
    payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Gift Card", "Crypto"])
with col12:
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Documentary"])

st.divider()

# Center Predict Button
center_btn_col1, center_btn_col2, center_btn_col3 = st.columns([2, 1, 2])
with center_btn_col2:
    predict_clicked = st.button("Predict", use_container_width=True, type="primary")

# ---------------------------------------------------------
# PREDICTION ANALYSIS RESULTS
# ---------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'subscription_type': subscription_type,
        'watch_hours': watch_hours,
        'last_login_days': last_login_days,
        'region': region,
        'device': device,
        'monthly_fee': monthly_fee,
        'payment_method': payment_method,
        'number_of_profiles': number_of_profiles,
        'avg_watch_time_per_day': avg_watch_time_per_day,
        'favorite_genre': favorite_genre
    }])

    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Results")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **Status: High Risk of Churn!**")
            st.metric(label="Churn Probability", value=f"{probabilities[1]*100:.2f}%")
        else:
            st.success("✅ **Status: Customer Retained (Low Churn Risk)**")
            st.metric(label="Retention Probability", value=f"{probabilities[0]*100:.2f}%")

    with res_col2:
        st.write("**Probability Breakdown**")
        prob_df = pd.DataFrame({
            'Outcome': ['Retained', 'Churned'],
            'Probability (%)': [probabilities[0]*100, probabilities[1]*100]
        })
        st.bar_chart(prob_df.set_index('Outcome'))
