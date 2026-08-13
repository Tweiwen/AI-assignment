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

st.title("🎬 Netflix Customer Churn Prediction System")
st.markdown("Predict customer churn risk using Machine Learning (Random Forest).")

# Cache model training so it runs instantly after initial load
@st.cache_resource
def train_model():
    # Load dataset from repository path
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

# Sidebar input controls
st.sidebar.header("📋 Customer Profile Input")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
watch_hours = st.sidebar.number_input("Total Watch Hours", min_value=0.0, max_value=500.0, value=25.0)
last_login_days = st.sidebar.slider("Days Since Last Login", min_value=0, max_value=60, value=10)
region = st.sidebar.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
device = st.sidebar.selectbox("Device Used", ["TV", "Mobile", "Desktop", "Tablet"])
monthly_fee = st.sidebar.number_input("Monthly Fee ($)", min_value=5.0, max_value=30.0, value=13.99)
payment_method = st.sidebar.selectbox("Payment Method", ["Credit Card", "PayPal", "Gift Card", "Crypto"])
number_of_profiles = st.sidebar.slider("Number of Profiles", min_value=1, max_value=5, value=2)
avg_watch_time_per_day = st.sidebar.number_input("Avg Daily Watch Time (Hours)", min_value=0.0, max_value=24.0, value=1.5)
favorite_genre = st.sidebar.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Documentary"])

# Perform Prediction
if st.sidebar.button("🔍 Predict Churn Status"):
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

    st.subheader("📊 Prediction Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ **Status: High Risk of Churn!**")
            st.metric(label="Churn Probability", value=f"{probabilities[1]*100:.2f}%")
        else:
            st.success("✅ **Status: Customer Retained (Low Churn Risk)**")
            st.metric(label="Retention Probability", value=f"{probabilities[0]*100:.2f}%")

    with col2:
        st.write("**Probability Breakdown**")
        prob_df = pd.DataFrame({
            'Outcome': ['Retained', 'Churned'],
            'Probability (%)': [probabilities[0]*100, probabilities[1]*100]
        })
        st.bar_chart(prob_df.set_index('Outcome'))
