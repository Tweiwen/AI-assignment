import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from model.utils import load_data, prepare_train_test_data, TARGET
from model.knn import train_knn_model, evaluate_knn, predict_knn
from model.lr import train_lr_model, evaluate_lr, predict_lr
from model.rd import train_rf_model, evaluate_rf, predict_rf

# Helper to load local logo as base64
@st.cache_data
def get_logo_base64(path="Netflix_Logo.png"):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Netflix Churn Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix White/Light Theme aesthetic with Darker Netflix Red accents (#B81D24)
st.markdown("""
<style>
    /* Light Theme Core Colors */
    .stApp {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    
    /* Title header styling */
    h1 {
        color: #B81D24 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Tab Header Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #E9ECEF;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 6px 6px 0px 0px;
        color: #374151;
        font-weight: 600;
        padding: 10px 20px;
        border-bottom: 3px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #B81D24 !important;
        border-bottom: 3px solid #B81D24 !important;
        box-shadow: 0px -2px 8px rgba(0, 0, 0, 0.04);
    }

    /* Form & Container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E9ECEF;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA & MODEL LOADING
# ==========================================
@st.cache_data
def get_raw_data():
    return load_data('netflix_customer_churn.csv')

@st.cache_resource
def get_trained_models(df):
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    knn_model = train_knn_model(X_train, y_train)
    lr_model = train_lr_model(X_train, y_train)
    rf_model = train_rf_model(X_train, y_train)
    
    knn_metrics = evaluate_knn(knn_model, X_test, y_test)
    lr_metrics = evaluate_lr(lr_model, X_test, y_test)
    rf_metrics = evaluate_rf(rf_model, X_test, y_test)
    
    return {
        'KNN': {'model': knn_model, 'metrics': knn_metrics},
        'Logistic Regression': {'model': lr_model, 'metrics': lr_metrics},
        'Random Forest': {'model': rf_model, 'metrics': rf_metrics},
        'data_split': (X_train, X_test, y_train, y_test)
    }

raw_df = get_raw_data()
model_suite = get_trained_models(raw_df)

# Main Application Title Banner with Netflix Logo
logo_b64 = get_logo_base64("Netflix_Logo.png")
if logo_b64:
    logo_img_html = f'<img src="data:image/png;base64,{logo_b64}" height="55" style="object-fit: contain;" />'
else:
    logo_img_html = ''

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; padding-bottom: 5px;">
    {logo_img_html}
    <h1 style="color: #B81D24; margin: 0; font-weight: 800; font-size: 2.2rem; line-height: 1.2;">Netflix Churn Analytics & Prediction</h1>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR FILTER BAR
# ==========================================
st.sidebar.markdown("## Dataset Filter Bar")
st.sidebar.markdown("<small style='color: #666;'>Filter customer records across all chart visualizations</small>", unsafe_allow_html=True)

# Filter: Subscription Type
sub_types = sorted(raw_df['subscription_type'].unique().tolist())
selected_subs = st.sidebar.multiselect(
    "Subscription Plan",
    options=sub_types,
    default=sub_types
)

# Filter: Region
regions = sorted(raw_df['region'].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=regions,
    default=regions
)

# Filter: Gender
genders = sorted(raw_df['gender'].unique().tolist())
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=genders,
    default=genders
)

# Filter: Payment Method
payments = sorted(raw_df['payment_method'].unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment Method",
    options=payments,
    default=payments
)

# Filter: Favorite Genre
genres = sorted(raw_df['favorite_genre'].unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Favorite Genre",
    options=genres,
    default=genres
)

# Filter: Age Range
min_age, max_age = int(raw_df['age'].min()), int(raw_df['age'].max())
selected_age = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Filter: Watch Hours Range
min_wh, max_wh = float(raw_df['watch_hours'].min()), float(raw_df['watch_hours'].max())
selected_wh = st.sidebar.slider(
    "Watch Hours Range",
    min_value=round(min_wh, 1),
    max_value=round(max_wh, 1),
    value=(round(min_wh, 1), round(max_wh, 1))
)

# Filter: Inactivity Days (last_login_days) Range
min_days, max_days = int(raw_df['last_login_days'].min()), int(raw_df['last_login_days'].max())
selected_days = st.sidebar.slider(
    "Inactivity Days (last_login_days)",
    min_value=min_days,
    max_value=max_days,
    value=(min_days, max_days)
)

# Apply filters to dataset
filtered_df = raw_df[
    (raw_df['subscription_type'].isin(selected_subs)) &
    (raw_df['region'].isin(selected_regions)) &
    (raw_df['gender'].isin(selected_genders)) &
    (raw_df['payment_method'].isin(selected_payments)) &
    (raw_df['favorite_genre'].isin(selected_genres)) &
    (raw_df['age'].between(selected_age[0], selected_age[1])) &
    (raw_df['watch_hours'].between(selected_wh[0], selected_wh[1])) &
    (raw_df['last_login_days'].between(selected_days[0], selected_days[1]))
]


# ==========================================
# MAIN TAB NAVIGATION
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer",
    "Visualisation Dashboard",
    "Model Comparison",
    "Predictor"
])


# ==========================================
# TAB 1: DATA EXPLORER
# ==========================================
with tab1:
    st.markdown("### Dataset Overview & Breakdown")
    
    if filtered_df.empty:
        st.warning("⚠️ No records match the selected sidebar filters. Please broaden your filter criteria.")
    else:
        total_records = len(filtered_df)
        churn_count = filtered_df['churned'].sum()
        retained_count = total_records - churn_count
        churn_rate = (churn_count / total_records * 100) if total_records > 0 else 0
        
        basic_count = len(filtered_df[filtered_df['subscription_type'] == 'Basic'])
        std_count = len(filtered_df[filtered_df['subscription_type'] == 'Standard'])
        prem_count = len(filtered_df[filtered_df['subscription_type'] == 'Premium'])

        # Total Records & Target Variable text summary with Target Variable Pie Chart
        col_text, col_target_pie = st.columns([1.2, 1])
        with col_text:
            st.markdown(f"**Total Records**: {total_records:,} fully populated records (0 missing values, 0 duplicates).")
            st.markdown("**Total Features**: 14 columns (1 ID, 1 Target, 12 Predictors)")
            st.markdown("**Target Variable**: Churned (1 = Churned, 0 = Retained)")
        
        with col_target_pie:
            target_pie_df = pd.DataFrame([
                {"Status": "Churned", "Count": churn_count},
                {"Status": "Retained", "Count": retained_count}
            ])
            fig_target_pie = px.pie(
                target_pie_df,
                names='Status',
                values='Count',
                title="Target Variable (Churn Breakdown)",
                color='Status',
                color_discrete_map={'Churned': '#B81D24', 'Retained': '#221F1F'},
                hole=0.4
            )
            fig_target_pie.update_layout(template="plotly_white", height=200, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_target_pie, use_container_width=True)

        st.markdown("---")
        
        # Subscription Breakdown in Table
        st.markdown("#### Subscription Breakdown Summary")
        sub_df = pd.DataFrame([
            {"Plan": "Basic", "Monthly Fee": "$8.99", "Count": basic_count, "Percentage": f"{(basic_count/total_records*100):.1f}%" if total_records else "0%"},
            {"Plan": "Standard", "Monthly Fee": "$13.99", "Count": std_count, "Percentage": f"{(std_count/total_records*100):.1f}%" if total_records else "0%"},
            {"Plan": "Premium", "Monthly Fee": "$17.99", "Count": prem_count, "Percentage": f"{(prem_count/total_records*100):.1f}%" if total_records else "0%"}
        ])
        
        col_table, col_sub_pie = st.columns([1, 1])
        with col_table:
            st.dataframe(sub_df, use_container_width=True, hide_index=True)
            
        with col_sub_pie:
            fig_sub_pie = px.pie(
                sub_df, names='Plan', values='Count',
                title="Subscription Plan Share",
                color='Plan',
                color_discrete_map={'Basic': '#B81D24', 'Standard': '#221F1F', 'Premium': '#574D4C'},
                hole=0.4
            )
            fig_sub_pie.update_layout(template="plotly_white", height=240, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sub_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Dataset Preview")
        
        # Rows to view control
        row_options = [5, 10, 20, 50, 100, "All"]
        selected_rows = st.selectbox("Rows to view:", options=row_options, index=2)
        
        # Dataset without customer_id
        preview_df = filtered_df.drop(columns=['customer_id'], errors='ignore')
        
        if selected_rows != "All":
            preview_df = preview_df.head(int(selected_rows))
            
        st.dataframe(preview_df, use_container_width=True, height=350)


# ==========================================
# TAB 2: VISUALISATION DASHBOARD
# ==========================================
with tab2:
    st.markdown("### Visualisation Dashboard")
    st.markdown("<small style='color: #666;'>Visualizing churn risk factors using filtered customer data</small>", unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("⚠️ No records match the current sidebar filter selection. Please adjust your filters.")
    else:
        grid_row1_col1, grid_row1_col2 = st.columns(2)
        
        # GRAPH 1: Subscription Type vs. Churn Rate (Vertical Bar Chart)
        with grid_row1_col1:
            st.markdown("#### 1. Subscription Type vs. Churn Rate")
            g1_data = filtered_df.groupby('subscription_type')['churned'].agg(
                Total='count',
                Churned='sum',
                ChurnRate=lambda x: (x.sum() / x.count()) * 100
            ).reset_index()
            
            # Ensure fixed order Basic, Standard, Premium
            plan_order = {'Basic': 1, 'Standard': 2, 'Premium': 3}
            g1_data['order'] = g1_data['subscription_type'].map(plan_order)
            g1_data = g1_data.sort_values('order')
            
            fig1 = px.bar(
                g1_data,
                x='subscription_type',
                y='ChurnRate',
                text=g1_data['ChurnRate'].apply(lambda val: f"{val:.1f}%"),
                color='subscription_type',
                color_discrete_map={'Basic': '#B81D24', 'Standard': '#221F1F', 'Premium': '#574D4C'},
                labels={'subscription_type': 'Subscription Type', 'ChurnRate': 'Churn Rate (%)'},
                title="Churn Rate by Subscription Plan"
            )
            fig1.update_traces(textposition='outside')
            fig1.update_layout(
                template="plotly_white",
                yaxis=dict(range=[0, min(100, max(g1_data['ChurnRate'].max() + 15, 20))]),
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig1, use_container_width=True)

        # GRAPH 2: Watch Hours Distribution by Churn Status (Comparative Box Plot)
        with grid_row1_col2:
            st.markdown("#### 2. Watch Hours Distribution by Churn Status")
            g2_df = filtered_df.copy()
            g2_df['Churn_Label'] = g2_df['churned'].map({0: 'Retained', 1: 'Churned'})
            
            fig2 = px.box(
                g2_df,
                x='Churn_Label',
                y='watch_hours',
                color='Churn_Label',
                points="outliers",
                color_discrete_map={'Retained': '#221F1F', 'Churned': '#B81D24'},
                labels={'Churn_Label': 'Churn Status', 'watch_hours': 'Watch Hours'},
                title="Watch Hours Distribution by Churn Status"
            )
            fig2.update_layout(
                template="plotly_white",
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        grid_row2_col1, grid_row2_col2 = st.columns(2)

        # GRAPH 3: Number of Profiles vs. Churn Rate (Grouped Bar Chart)
        with grid_row2_col1:
            st.markdown("#### 3. Number of Profiles vs. Churn Rate")
            g3_df = filtered_df.groupby(['number_of_profiles', 'churned']).size().reset_index(name='count')
            g3_df['Churn Status'] = g3_df['churned'].map({0: 'Retained', 1: 'Churned'})
            
            fig3 = px.bar(
                g3_df,
                x='number_of_profiles',
                y='count',
                color='Churn Status',
                barmode='group',
                color_discrete_map={'Retained': '#221F1F', 'Churned': '#B81D24'},
                labels={'number_of_profiles': 'Number of Profiles', 'count': 'Customer Count'},
                title="Profiles vs. Churn Breakdown"
            )
            fig3.update_layout(
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig3, use_container_width=True)

        # GRAPH 4: Inactivity Days vs. Churn Probability Trend Line Chart
        with grid_row2_col2:
            st.markdown("#### 4. Inactivity Days vs. Churn Probability Trend")
            g4_df = filtered_df.copy()
            g4_df['inactivity_bin'] = (g4_df['last_login_days'] // 5) * 5
            trend_df = g4_df.groupby('inactivity_bin')['churned'].agg(
                Total='count',
                ChurnRate=lambda x: (x.sum() / x.count()) * 100
            ).reset_index()
            
            fig4 = px.line(
                trend_df,
                x='inactivity_bin',
                y='ChurnRate',
                markers=True,
                labels={'inactivity_bin': 'Inactivity Days (last_login_days)', 'ChurnRate': 'Churn Probability (%)'},
                title="Inactivity Days vs. Churn Probability"
            )
            fig4.update_traces(line=dict(color='#B81D24', width=3), marker=dict(size=8, color='#B81D24'))
            fig4.update_layout(
                template="plotly_white",
                yaxis=dict(range=[0, min(100, max(trend_df['ChurnRate'].max() + 10, 20))]),
                height=380
            )
            st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# TAB 3: MODEL COMPARISON
# ==========================================
with tab3:
    st.markdown("### Machine Learning Model Comparison")
    st.markdown("<small style='color: #666;'>Evaluating KNN, Random Forest, and Logistic Regression models on held-out test data (20% split)</small>", unsafe_allow_html=True)
    
    # Metrics Table Construction (Decimals Only)
    metrics_summary = []
    for model_name in ['KNN', 'Logistic Regression', 'Random Forest']:
        m = model_suite[model_name]['metrics']
        metrics_summary.append({
            'Model': model_name,
            'Accuracy': f"{m['Accuracy']:.4f}",
            'Precision': f"{m['Precision']:.4f}",
            'Recall': f"{m['Recall']:.4f}",
            'F1-Score': f"{m['F1-Score']:.4f}"
        })
        
    metrics_df = pd.DataFrame(metrics_summary)
    
    st.markdown("#### Model Evaluation Performance Table")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### Actual vs Predicted Performance Visualizations")
    
    col_cm, col_pred_compare = st.columns([1, 1])
    
    with col_cm:
        st.markdown("##### Confusion Matrices")
        selected_cm_model = st.selectbox("Select Model for Confusion Matrix:", options=['Random Forest', 'Logistic Regression', 'KNN'])
        
        cm = model_suite[selected_cm_model]['metrics']['Confusion Matrix']
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Retained (0)', 'Churned (1)'],
            y=['Retained (0)', 'Churned (1)'],
            text_auto=True,
            color_continuous_scale='Reds',
            title=f"Confusion Matrix: {selected_cm_model}"
        )
        fig_cm.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_pred_compare:
        st.markdown("##### Actual vs Predicted Churn Count Comparison")
        X_train, X_test, y_train, y_test = model_suite['data_split']
        
        pred_comp_data = []
        # Actual count
        pred_comp_data.append({'Model': 'Actual (Ground Truth)', 'Status': 'Retained (0)', 'Count': (y_test == 0).sum()})
        pred_comp_data.append({'Model': 'Actual (Ground Truth)', 'Status': 'Churned (1)', 'Count': (y_test == 1).sum()})
        
        # Predictions
        for m_name in ['KNN', 'Logistic Regression', 'Random Forest']:
            mdl = model_suite[m_name]['model']
            y_pred = mdl.predict(X_test)
            pred_comp_data.append({'Model': m_name, 'Status': 'Retained (0)', 'Count': int((y_pred == 0).sum())})
            pred_comp_data.append({'Model': m_name, 'Status': 'Churned (1)', 'Count': int((y_pred == 1).sum())})
            
        pred_comp_df = pd.DataFrame(pred_comp_data)
        
        fig_pred = px.bar(
            pred_comp_df,
            x='Model',
            y='Count',
            color='Status',
            barmode='group',
            color_discrete_map={'Retained (0)': '#221F1F', 'Churned (1)': '#B81D24'},
            title="Actual vs Predicted Churn Distribution"
        )
        fig_pred.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_pred, use_container_width=True)


# ==========================================
# TAB 4: PREDICTOR
# ==========================================
with tab4:
    st.markdown("### Individual Customer Churn Predictor")
    st.markdown("<small style='color: #666;'>Input custom customer features to simulate and predict churn risk in real-time</small>", unsafe_allow_html=True)
    
    col_input_left, col_input_right = st.columns(2)
    
    with col_input_left:
        st.markdown("#### Demographics & Plan Details")
        in_age = st.number_input("Age", min_value=18, max_value=100, value=35)
        in_gender = st.selectbox("Gender", options=raw_df['gender'].unique())
        in_sub = st.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
        
        fee_mapping = {'Basic': 8.99, 'Standard': 13.99, 'Premium': 17.99}
        in_monthly_fee = fee_mapping[in_sub]
        st.caption(f"Monthly Fee auto-set to: **${in_monthly_fee}**")
        
        in_region = st.selectbox("Region", options=raw_df['region'].unique())
        in_payment = st.selectbox("Payment Method", options=raw_df['payment_method'].unique())

    with col_input_right:
        st.markdown("#### Usage & Engagement Metrics")
        in_watch_hours = st.number_input("Watch Hours (Total)", min_value=0.0, max_value=200.0, value=12.5, step=0.5)
        in_last_login = st.number_input("Inactivity Days (last_login_days)", min_value=0, max_value=180, value=15)
        in_device = st.selectbox("Primary Device", options=raw_df['device'].unique())
        in_profiles = st.slider("Number of Profiles", min_value=1, max_value=5, value=2)
        in_avg_daily_watch = st.number_input("Avg Daily Watch Time (Hours)", min_value=0.0, max_value=24.0, value=1.2, step=0.1)
        in_genre = st.selectbox("Favorite Genre", options=raw_df['favorite_genre'].unique())

    st.markdown("---")
    col_model_select, col_predict_btn = st.columns([2, 1])
    
    with col_model_select:
        selected_pred_model = st.selectbox("Choose Prediction Algorithm:", options=['Random Forest', 'Logistic Regression', 'KNN'])
        
    with col_predict_btn:
        st.write("") # vertical spacing
        st.write("")
        predict_submitted = st.button("Predict Churn Risk", use_container_width=True, type="primary")

    if predict_submitted:
        # Prepare single record input DataFrame
        input_data = pd.DataFrame([{
            'age': in_age,
            'gender': in_gender,
            'subscription_type': in_sub,
            'watch_hours': in_watch_hours,
            'last_login_days': in_last_login,
            'region': in_region,
            'device': in_device,
            'monthly_fee': in_monthly_fee,
            'payment_method': in_payment,
            'number_of_profiles': in_profiles,
            'avg_watch_time_per_day': in_avg_daily_watch,
            'favorite_genre': in_genre
        }])
        
        mdl = model_suite[selected_pred_model]['model']
        pred_label, pred_proba = mdl.predict(input_data)[0], mdl.predict_proba(input_data)[0][1]
        
        churn_pct = pred_proba * 100
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if pred_label == 1:
                st.error(f"⚠️ **HIGH RISK OF CHURN DETECTED** ({selected_pred_model})")
                st.markdown(f"The customer is predicted to **CHURN** with **{churn_pct:.1f}%** probability.")
            else:
                st.success(f"✅ **CUSTOMER LIKELY TO STAY** ({selected_pred_model})")
                st.markdown(f"The customer is predicted to **RETAIN** with **{100-churn_pct:.1f}%** confidence.")
                
            # Risk Level Badge
            if churn_pct >= 65:
                risk_badge = "High Risk (> 65%)"
            elif churn_pct >= 35:
                risk_badge = "Medium Risk (35% - 65%)"
            else:
                risk_badge = "Low Risk (< 35%)"
                
            st.info(f"**Risk Level Assessment**: {risk_badge}")

        with res_col2:
            # Compact Probability Gauge with small margins and font fitting
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_pct,
                domain={'x': [0.05, 0.95], 'y': [0, 0.9]},
                title={'text': "Churn Probability (%)", 'font': {'size': 13, 'color': '#111111'}},
                number={'suffix': "%", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333333", 'tickfont': {'size': 10}},
                    'bar': {'color': "#B81D24" if churn_pct >= 50 else "#221F1F"},
                    'bgcolor': "#FFFFFF",
                    'bordercolor': "#CCCCCC",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(34, 31, 31, 0.1)'},
                        {'range': [35, 65], 'color': 'rgba(241, 196, 15, 0.15)'},
                        {'range': [65, 100], 'color': 'rgba(184, 29, 36, 0.15)'}
                    ]
                }
            ))
            fig_gauge.update_layout(template="plotly_white", height=170, margin=dict(l=15, r=15, t=35, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)