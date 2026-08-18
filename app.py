# ==============================================================================
# BMCS2074 ARTIFICIAL INTELLIGENCE - ASSIGNMENT PROTOTYPE
# TITLE: NETFLIX CUSTOMER CHURN PREDICTION (SUPERVISED MACHINE LEARNING)
# ==============================================================================

import os
import time
import tracemalloc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & HEADER
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide"
)

st.caption("BMCS2074 ARTIFICIAL INTELLIGENCE - ASSIGNMENT PROTOTYPE")
st.title("🎬 Netflix Customer Churn Prediction System")
st.markdown("### Supervised Machine Learning Benchmark & Interactive Churn Predictor")
st.divider()

# ---------------------------------------------------------
# FAST CACHED TRAINING & EVALUATION LOGIC
# ---------------------------------------------------------
@st.cache_resource
def load_and_evaluate_models():
    # File path resolution across directory layouts
    possible_paths = [
        '03_Dataset/netflix_customer_churn.csv',
        'netflix_customer_churn.csv'
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        st.error("❌ Error: Dataset file 'netflix_customer_churn.csv' missing.")
        st.stop()

    # Step 1: Data Loading & Preprocessing
    raw_df = pd.read_csv(dataset_path)
    df = raw_df.copy()

    # Drop primary identifiers not useful for machine learning
    if "customer_id" in df.columns:
        df.drop(columns=["customer_id"], inplace=True)

    # Check and remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)

    # Handle Missing Values (Imputation)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if 'churned' in num_cols:
        num_cols.remove('churned')
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    X = df.drop(columns=["churned"])
    y = df["churned"]

    # Preprocessing Pipeline (StandardScaler + OneHotEncoder)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fast & Optimized Model Algorithms with Best Hyperparameters
    model_configs = {
        'KNN': {
            'name': 'K-Nearest Neighbors (KNN)',
            'clf': KNeighborsClassifier(n_neighbors=5),
            'best_params': 'n_neighbors=5',
            'cmap': 'Blues'
        },
        'SVM': {
            'name': 'Support Vector Machine (SVM)',
            'clf': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
            'best_params': 'kernel=rbf, C=1.0',
            'cmap': 'Greens'
        },
        'Random Forest': {
            'name': 'Random Forest Ensemble',
            'clf': RandomForestClassifier(n_estimators=100, random_state=42),
            'best_params': 'n_estimators=100, max_depth=None',
            'cmap': 'Oranges'
        }
    }

    results = []
    trained_pipelines = {}
    evaluation_details = {}

    for key, config in model_configs.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['clf'])
        ])

        tracemalloc.start()
        start_time = time.perf_counter()

        pipe.fit(X_train, y_train)

        exec_time = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"])
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'Algorithm': config['name'],
            'Best Hyperparameters': config['best_params'],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'Execution Time (s)': round(exec_time, 4),
            'Peak Memory (KB)': round(peak_mem / 1024, 2)
        })

        trained_pipelines[key] = pipe
        evaluation_details[key] = {
            'name': config['name'],
            'report': report,
            'cm': cm,
            'cmap': config['cmap'],
            'best_params': config['best_params']
        }

    # Save best trained model pipeline (Random Forest)
    os.makedirs('04_Trained_Model', exist_ok=True)
    joblib.dump(trained_pipelines['Random Forest'], '04_Trained_Model/netflix_churn_pipeline.pkl')

    res_df = pd.DataFrame(results)

    # Save evaluation summary to model_comparison_results.csv
    export_df = res_df[['Algorithm', 'Best Hyperparameters', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Execution Time (s)', 'Peak Memory (KB)']]
    export_df.to_csv('model_comparison_results.csv', index=False)

    # Feature Importance Analysis for Random Forest
    rf_pipeline = trained_pipelines['Random Forest']
    rf_clf = rf_pipeline.named_steps['classifier']
    fitted_preprocessor = rf_pipeline.named_steps['preprocessor']
    feature_names = fitted_preprocessor.get_feature_names_out()
    cleaned_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

    imp_df = pd.DataFrame({
        'Feature': cleaned_feature_names,
        'Importance': rf_clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    meta = {
        'initial_rows': initial_rows,
        'duplicates_removed': duplicates_removed,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    return trained_pipelines, res_df, evaluation_details, imp_df, meta

with st.spinner("Initializing system & loading ML models..."):
    trained_pipelines, comparison_df, eval_details, feature_importance_df, metadata = load_and_evaluate_models()
    best_pipeline = trained_pipelines['Random Forest']

# ---------------------------------------------------------
# STATE MANAGEMENT & MONTHLY FEE DEFAULTS
# ---------------------------------------------------------
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False

FEE_DEFAULTS = {
    "Basic": 8.90,
    "Standard": 10.90,
    "Premium": 20.90
}

def update_monthly_fee():
    plan = st.session_state.get('subscription_type', 'Basic')
    st.session_state.monthly_fee = FEE_DEFAULTS.get(plan, 8.90)

if 'monthly_fee' not in st.session_state:
    st.session_state.monthly_fee = FEE_DEFAULTS["Basic"]

# ---------------------------------------------------------
# HOMEPAGE TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 Customer Details Input", 
    "📊 Prediction Results", 
    "📈 Model Comparison & Benchmark Suite"
])

# =========================================================
# TAB 1: INPUT FORM
# =========================================================
with tab1:
    st.subheader("Customer Profile & Subscription Parameters")
    st.markdown("Specify customer demographic and usage features to predict churn risk.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        subscription_type = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"],
            key="subscription_type",
            on_change=update_monthly_fee
        )
    with col4:
        monthly_fee = st.number_input(
            "Monthly Fee ($)",
            min_value=5.0,
            max_value=30.0,
            step=0.10,
            format="%.2f",
            key="monthly_fee"
        )

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
            st.success("✅ Prediction complete! Open the **'📊 Prediction Results'** tab above.")

# =========================================================
# TAB 2: PREDICTION RESULTS
# =========================================================
with tab2:
    st.subheader("Model Output & Risk Assessment")

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
                st.success("✅ **Status: Customer Retained (Low Risk)**")
                st.metric(label="Retention Probability", value=f"{probabilities[0]*100:.2f}%")

            st.divider()
            st.markdown("### Submitted Profile Summary")
            st.dataframe(st.session_state.last_input.T, use_container_width=True)

        with res_col2:
            st.markdown("### Outcome Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Retained', 'Churned'],
                'Probability (%)': [probabilities[0]*100, probabilities[1]*100]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
    else:
        st.info("👈 Enter profile details in the **'📋 Customer Details Input'** tab and click **'Predict Churn Status'** first.")

# =========================================================
# TAB 3: MODEL COMPARISON & BENCHMARK SUITE
# =========================================================
with tab3:
    st.subheader("Model Evaluation & Algorithm Comparison")
    st.markdown("Comprehensive benchmarking with hyperparameter evaluation, execution time logging, and resource profiling.")

    # Data Preprocessing Summary Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(label="Total Raw Records", value=f"{metadata['initial_rows']:,}")
    with mcol2:
        st.metric(label="Duplicates Removed", value=f"{metadata['duplicates_removed']:,}")
    with mcol3:
        st.metric(label="Training Set Size (80%)", value=f"{metadata['train_samples']:,}")
    with mcol4:
        st.metric(label="Testing Set Size (20%)", value=f"{metadata['test_samples']:,}")

    st.divider()

    # Comparative Results Table
    st.markdown("### Final Comparative Summary")
    display_df = comparison_df[['Algorithm', 'Best Hyperparameters', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Execution Time (s)', 'Peak Memory (KB)']]
    st.dataframe(
        display_df.style.highlight_max(axis=0, subset=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']),
        use_container_width=True
    )

    st.caption("✅ Evaluation summary automatically saved to `model_comparison_results.csv`.")

    st.divider()

    # Metric Comparison Visualization
    st.markdown("### Comparative Metric Scores")
    chart_df = comparison_df.set_index('Algorithm')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    st.bar_chart(chart_df, height=350)

    st.divider()

    # Confusion Matrices Section
    st.markdown("### Confusion Matrix Visualizations")
    cm_col1, cm_col2, cm_col3 = st.columns(3)

    for idx, (key, details) in enumerate(eval_details.items()):
        col_target = [cm_col1, cm_col2, cm_col3][idx]
        with col_target:
            st.markdown(f"#### {details['name']}")
            st.caption(f"**Best Params:** `{details['best_params']}`")

            fig, ax = plt.subplots(figsize=(4, 3.5))
            disp = ConfusionMatrixDisplay(confusion_matrix=details['cm'], display_labels=["Retained", "Churned"])
            disp.plot(cmap=details['cmap'], ax=ax, colorbar=False)
            plt.title(f"{key} Confusion Matrix", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

    st.divider()

    # Detailed Classification Reports Section
    st.markdown("### Detailed Classification Reports")
    rep_col1, rep_col2, rep_col3 = st.columns(3)

    for idx, (key, details) in enumerate(eval_details.items()):
        col_target = [rep_col1, rep_col2, rep_col3][idx]
        with col_target:
            with st.expander(f"📑 {details['name']} Classification Report", expanded=True):
                st.code(details['report'], language="text")

    st.divider()

    # Random Forest Feature Importance Analysis
    st.markdown("### Random Forest Feature Importance Analysis")
    st.markdown("Relative significance of attributes in predicting customer churn.")

    fig, ax = plt.subplots(figsize=(10, 5))
    top_features = feature_importance_df.head(10)
    sns.barplot(x="Importance", y="Feature", data=top_features, hue="Feature", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 10 Feature Importances (Random Forest Ensemble)", fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Attribute")
    plt.tight_layout()
    st.pyplot(fig)
