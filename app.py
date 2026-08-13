import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide"
)

# Header Section
st.title("🎬 Netflix Customer Churn Prediction System")
st.markdown("Compare machine learning models and predict customer churn risk.")
st.divider()

# Cache model training and comparison
@st.cache_resource
def train_and_evaluate_models():
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            'Model': name,
            'Accuracy (%)': round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision (%)': round(precision_score(y_test, y_pred) * 100, 2),
            'Recall (%)': round(recall_score(y_test, y_pred) * 100, 2),
            'F1-Score (%)': round(f1_score(y_test, y_pred) * 100, 2)
        })
        trained_pipelines[name] = pipeline

    results_df = pd.DataFrame(results)
    return trained_pipelines, results_df

# Train / load model and benchmarks
with st.spinner("Training models & calculating performance comparison..."):
    trained_pipelines, comparison_df = train_and_evaluate_models()
    best_pipeline = trained_pipelines['Random Forest']

# Session State for prediction tracking
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False

# Three Homepage Tabs
tab1, tab2, tab3 = st.tabs([
    "📋 Customer Details Input", 
    "📊 Prediction Results", 
    "📈 Model Comparison (KNN vs SVM vs RF)"
])

# ---------------------------------------------------------
# TAB 1: INPUT FORM
# ---------------------------------------------------------
with tab1:
    st.subheader("Enter Customer Profile Features")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    with col4:
        monthly_fee = st.number_input("Monthly Fee ($)", min_value=5.0, max_value=30.0, value=13.99)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        watch_hours = st.number_input("Total Watch Hours", min_value=0.0, max_value=500.0, value=25.0)
    with col6:
        avg_watch_time_per_day = st.number_input("Avg Daily Watch Time (Hrs)", min_value=0.0, max_value=24.0, value=1.5)
    with col7:
        last_login_days = st.slider("Days Since Last Login", min_value=0, max_value=60, value=10)
    with col8:
        number_of_profiles = st.slider("Number of Profiles", min_value=1, max_value=5, value=2)

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

    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
    with btn_col2:
        if st.button("🔍 Predict Churn Status", use_container_width=True, type="primary"):
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

            st.session_state.prediction = best_pipeline.predict(input_df)[0]
            st.session_state.probabilities = best_pipeline.predict_proba(input_df)[0]
            st.session_state.has_predicted = True
            st.session_state.last_input = input_df
            st.success("Prediction generated using **Random Forest**! Open the **'📊 Prediction Results'** tab above.")

# ---------------------------------------------------------
# TAB 2: PREDICTION RESULTS
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Output Analysis (Random Forest)")

    if st.session_state.has_predicted:
        prediction = st.session_state.prediction
        probabilities = st.session_state.probabilities

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("### Risk Status")
            if prediction == 1:
                st.error("⚠️ **Status: High Risk of Churn!**")
                st.metric(label="Churn Probability", value=f"{probabilities[1]*100:.2f}%")
            else:
                st.success("✅ **Status: Customer Retained (Low Churn Risk)**")
                st.metric(label="Retention Probability", value=f"{probabilities[0]*100:.2f}%")

            st.divider()
            st.markdown("### Submitted Inputs Summary")
            st.dataframe(st.session_state.last_input.T, use_container_width=True)

        with res_col2:
            st.markdown("### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['Retained', 'Churned'],
                'Probability (%)': [probabilities[0]*100, probabilities[1]*100]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
    else:
        st.info("👈 Please enter customer details in the **'📋 Customer Details Input'** tab and click **'Predict Churn Status'** first.")

# ---------------------------------------------------------
# TAB 3: MODEL COMPARISON (KNN vs SVM vs RANDOM FOREST)
# ---------------------------------------------------------
with tab3:
    st.subheader("Model Performance Comparison")
    st.markdown("Evaluation metrics calculated on the 20% hold-out test dataset.")

    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']), use_container_width=True)

    st.divider()
    st.markdown("### Accuracy Comparison Visual")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=comparison_df, x='Model', y='Accuracy (%)', palette='Set2', ax=ax)
    ax.set_ylim(70, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    st.pyplot(fig)
