import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Netflix Churn Predictor", layout="wide")

# Load Preprocessor and Models
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessing_pipeline.pkl')
    rf_model = joblib.load('models/model_random.pkl')
    svm_model = joblib.load('models/model_svm.pkl')
    knn_model = joblib.load('models/model_knn.pkl')
    return preprocessor, {"Random Forest": rf_model, "SVM": svm_model, "KNN": knn_model}

try:
    preprocessor, models = load_artifacts()
except Exception as e:
    st.error("Error loading model files. Please run `train_models.py` first.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", ["Dashboard", "Single Prediction", "Batch Prediction", "Model Performance"])

# MODULE 1: DASHBOARD
if page == "Dashboard":
    st.title("🎬 Netflix Customer Churn Analytics")
    st.markdown("Overview of customer dataset distributions and key metrics.")
    
    df = pd.read_csv('../03_Dataset/netflix_customer_churn.csv')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{(df['churned'].mean() * 100):.1f}%")
    col3.metric("Avg Watch Hours", f"{df['watch_hours'].mean():.1f} hrs")
    col4.metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")
    
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Churn Distribution by Subscription Type")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='subscription_type', hue='churned', palette='Set2', ax=ax)
        st.pyplot(fig)
        
    with col_right:
        st.subheader("Watch Time vs Last Login Days")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='last_login_days', y='watch_hours', hue='churned', alpha=0.6, ax=ax2)
        st.pyplot(fig2)

# MODULE 2: SINGLE PREDICTION
elif page == "Single Prediction":
    st.title("👤 Customer Churn Risk Evaluator")
    st.markdown("Enter customer details below to predict their likelihood of churning.")
    
    selected_model_name = st.selectbox("Select Classification Model:", list(models.keys()))
    selected_model = models[selected_model_name]
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            region = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
            
        with col2:
            watch_hours = st.number_input("Watch Hours (Total)", min_value=0.0, value=15.0)
            avg_daily_watch = st.number_input("Avg Watch Time / Day (hrs)", min_value=0.0, value=1.5)
            last_login = st.number_input("Last Login Days Ago", min_value=0, max_value=365, value=5)
            profiles = st.number_input("Number of Profiles", min_value=1, max_value=5, value=2)
            
        with col3:
            device = st.selectbox("Primary Device", ["Desktop", "Laptop", "Mobile", "Tablet", "TV"])
            monthly_fee = st.number_input("Monthly Fee ($)", min_value=0.0, value=11.99)
            payment = st.selectbox("Payment Method", ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"])
            genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])
            
        submit_btn = st.form_submit_button("Predict Churn Risk")
        
    if submit_btn:
        input_data = pd.DataFrame([{
            'age': age, 'gender': gender, 'subscription_type': sub_type, 'region': region,
            'watch_hours': watch_hours, 'avg_watch_time_per_day': avg_daily_watch,
            'last_login_days': last_login, 'number_of_profiles': profiles,
            'device': device, 'monthly_fee': monthly_fee, 'payment_method': payment,
            'favorite_genre': genre
        }])
        
        # Preprocess & Predict
        input_prep = preprocessor.transform(input_data)
        prediction = selected_model.predict(input_prep)[0]
        proba = selected_model.predict_proba(input_prep)[0][1] * 100
        
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk!** Confidence: **{proba:.2f}%**")
            st.warning("Recommendation: Offer a promotional discount or personalized content suggestions.")
        else:
            st.success(f"✅ **Low Churn Risk (Retained).** Retention Likelihood: **{(100 - proba):.2f}%**")

# MODULE 3: BATCH PREDICTION
elif page == "Batch Prediction":
    st.title("📁 Batch Customer Prediction")
    uploaded_file = st.file_uploader("Upload CSV file containing customer data", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Preview:", batch_df.head(3))
        
        model_choice = st.selectbox("Choose Model for Batch Processing:", list(models.keys()))
        
        if st.button("Run Batch Predictions"):
            try:
                X_batch = batch_df.drop(columns=['customer_id', 'churned'], errors='ignore')
                processed_batch = preprocessor.transform(X_batch)
                preds = models[model_choice].predict(processed_batch)
                
                batch_df['Predicted_Churn'] = preds
                st.success("Predictions complete!")
                st.dataframe(batch_df.head(10))
                
                # Download button
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Prediction Results CSV", csv_out, "churn_predictions.csv", "text/csv")
            except Exception as err:
                st.error(f"Error during processing: {err}")

# MODULE 4: MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("📊 Model Comparison & Evaluation")
    st.markdown("Comparison of evaluation metrics across group members' implementations.")
    
    try:
        metrics_df = pd.read_csv('model_comparison_results.csv')
        st.table(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='lightgreen'))
    except FileNotFoundError:
        st.warning("Metrics file not found. Ensure `train_models.py` has been executed.")
