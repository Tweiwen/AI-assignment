import base64
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import requests
from google import genai

from model.utils import load_data, prepare_train_test_data, TARGET, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from model.knn import train_knn_model, evaluate_knn, predict_knn
from model.lr import train_lr_model, evaluate_lr, predict_lr
from model.rf import train_rf_model, evaluate_rf, predict_rf

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
    page_title="Netflix Churn Prediction & Retention Support System",
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
# SESSION STATE INITIALISATION
# ==========================================
if 'selected_customer_id' not in st.session_state:
    st.session_state['selected_customer_id'] = None


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


# ==========================================
# SCORE ALL CUSTOMERS WITH RANDOM FOREST
# ==========================================
@st.cache_data
def score_all_customers(_rf_model, df):
    """Use the trained Random Forest pipeline to score every customer's churn probability."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_all = df[feature_cols]
    probas = _rf_model.predict_proba(X_all)[:, 1]
    
    scored = df.copy()
    scored['churn_probability'] = probas
    scored['churn_pct'] = (probas * 100).round(1)
    scored['risk_level'] = pd.cut(
        scored['churn_pct'],
        bins=[-1, 35, 65, 101],
        labels=['Low', 'Medium', 'High'],
        right=False
)
    scored = scored.sort_values('churn_probability', ascending=False).reset_index(drop=True)
    return scored

scored_df = score_all_customers(model_suite['Random Forest']['model'], raw_df)


# ==========================================
# HELPER: COMPUTE DATASET STATISTICS
# ==========================================
@st.cache_data
def get_dataset_stats(df):
    """Compute dataset-wide statistics used for engagement comparisons."""
    return {
        'avg_watch_hours': float(df['watch_hours'].mean()),
        'avg_inactivity': float(df['last_login_days'].mean()),
        'avg_daily_watch': float(df['avg_watch_time_per_day'].mean()),
        'avg_profiles': float(df['number_of_profiles'].mean()),
        'overall_churn_rate': float(df['churned'].mean() * 100),
    }

dataset_stats = get_dataset_stats(raw_df)


# ==========================================
# HELPER: GENERATE ENGAGEMENT SUMMARY
# ==========================================
def generate_engagement_summary(customer_row, stats):
    """
    Generate 2-4 simple factual observations comparing the customer's
    engagement metrics against dataset averages.
    Uses careful language — never claims causation.
    """
    observations = []

    # Inactivity comparison
    inactivity = customer_row['last_login_days']
    avg_inact = stats['avg_inactivity']

    if inactivity > avg_inact:
        observations.append(
            f"This customer has been inactive for "
            f"**<span style='color:red;'>{int(inactivity)}</span> days**, "
            f"compared with the dataset average of **{avg_inact:.0f} days**."
        )
    else:
        observations.append(
            f"This customer's inactivity period is "
            f"**<span style='color:red;'>{int(inactivity)}</span> days**, "
            f"which is at or below the dataset average of **{avg_inact:.0f} days**."
        )


    # Average daily watch time comparison
    avg_daily = customer_row['avg_watch_time_per_day']
    ds_avg_daily = stats['avg_daily_watch']

    if avg_daily < ds_avg_daily:
        observations.append(
            f"Average daily viewing is "
            f"**<span style='color:red;'>{avg_daily:.2f}</span> hours**, "
            f"compared with the dataset average of **{ds_avg_daily:.2f} hours**."
        )
    else:
        observations.append(
            f"Average daily viewing is "
            f"**<span style='color:red;'>{avg_daily:.2f}</span> hours**, "
            f"which is at or above the dataset average of **{ds_avg_daily:.2f} hours**."
        )


    # Total watch hours comparison
    watch_hours = customer_row['watch_hours']
    avg_wh = stats['avg_watch_hours']

    if watch_hours < avg_wh:
        observations.append(
            f"Total watch hours are "
            f"**<span style='color:red;'>{watch_hours:.1f}</span> hours**, "
            f"compared with the dataset average of **{avg_wh:.1f} hours**."
        )
    else:
        observations.append(
            f"Total watch hours are "
            f"**<span style='color:red;'>{watch_hours:.1f}</span> hours**, "
            f"which is at or above the dataset average of **{avg_wh:.1f} hours**."
        )


    # Number of profiles (only mention if notably low)
    profiles = customer_row['number_of_profiles']
    avg_prof = stats['avg_profiles']

    if profiles <= 1:
        observations.append(
            f"This customer uses only "
            f"**<span style='color:red;'>{int(profiles)}</span> profile**, "
            f"compared with the dataset average of **{avg_prof:.1f} profiles**."
        )

    return observations


# ==========================================
# HELPER: GENERATE RETENTION RECOMMENDATIONS
# ==========================================
def generate_recommendations(customer_row, risk_level, churn_pct, stats):
    """
    Generate rule-based retention strategy suggestions based on the
    customer's predicted risk level and account characteristics.
    These are decision-support suggestions, not ML model outputs.
    """
    recommendations = []
    risk_str = str(risk_level)

    if risk_str == 'High':
        # High risk + high inactivity
        if customer_row['last_login_days'] > stats['avg_inactivity']:
            recommendations.append({
                'action': '📧 Re-engagement Message & Retention Voucher',
                'detail': (
                    f"This customer has been inactive for {int(customer_row['last_login_days'])} days. "
                    f"Send a personalised re-engagement message and consider providing a limited-time "
                    f"retention voucher or subscription discount to encourage the customer to return."
                ),
                'priority': 'High'
            })

        # High risk + low viewing activity
        if customer_row['watch_hours'] < stats['avg_watch_hours'] or customer_row['avg_watch_time_per_day'] < stats['avg_daily_watch']:
            genre = customer_row['favorite_genre']
            recommendations.append({
                'action': f'🎬 Personalised Content & Retention Incentive',
                'detail': (
                    f"Provide personalised recommendations based on the customer's favourite genre "
                    f"(**{genre}**) and consider a limited-time retention incentive. For example, "
                    f"recommend newly available {genre} content together with a retention voucher."
                ),
                'priority': 'High'
            })

        # High risk + Basic plan → trial upgrade
        if customer_row['subscription_type'] == 'Basic':
            recommendations.append({
                'action': '⬆️ Temporary Free Trial Upgrade',
                'detail': (
                    "This high-risk customer is on the Basic plan ($8.99/month). Consider offering a "
                    "temporary free trial upgrade from Basic to Standard as a retention incentive to "
                    "demonstrate additional value and features."
                ),
                'priority': 'High'
            })

        # Premium customer + low usage → suggest downgrade
        if customer_row['subscription_type'] == 'Premium' and customer_row['watch_hours'] < stats['avg_watch_hours']:
            recommendations.append({
                'action': '💰 Review Subscription Plan',
                'detail': (
                    "This customer is on the Premium plan ($17.99/month) but has below-average viewing "
                    "activity. Consider recommending a lower-cost subscription plan if the customer is "
                    "not making sufficient use of the Premium plan, which may improve perceived value."
                ),
                'priority': 'High'
            })

        # General high-risk retention promotion (if few recs so far)
        if len(recommendations) < 2:
            recommendations.append({
                'action': '🎁 Targeted Retention Promotion',
                'detail': (
                    f"With a churn probability of {churn_pct:.1f}%, consider offering a targeted "
                    f"retention promotion such as a temporary subscription discount, bonus feature "
                    f"access, or loyalty reward to reduce churn risk."
                ),
                'priority': 'High'
            })

    elif risk_str == 'Medium':
        # Content recommendations
        genre = customer_row['favorite_genre']
        recommendations.append({
            'action': f'🎬 Personalised {genre} Content Recommendations',
            'detail': (
                f"Send personalised content recommendations based on the customer's favourite genre "
                f"(**{genre}**) to maintain engagement and increase viewing time."
            ),
            'priority': 'Medium'
        })

        if customer_row['last_login_days'] > stats['avg_inactivity']:
            recommendations.append({
                'action': '📧 Gentle Check-in Message',
                'detail': (
                    f"This customer has been inactive for {int(customer_row['last_login_days'])} days. "
                    f"A friendly check-in message with content updates may help maintain engagement "
                    f"before risk escalates."
                ),
                'priority': 'Medium'
            })

        # Small retention voucher consideration
        recommendations.append({
            'action': '🎁 Consider Small Retention Voucher',
            'detail': (
                "If customer engagement continues to decrease, consider providing a small retention "
                "voucher as an incentive to continue the subscription. Continue monitoring engagement trends."
            ),
            'priority': 'Medium'
        })

        # Premium medium-risk with low usage
        if customer_row['subscription_type'] == 'Premium' and customer_row['watch_hours'] < stats['avg_watch_hours']:
            recommendations.append({
                'action': '💰 Review Subscription Plan',
                'detail': (
                    "This customer is on the Premium plan but has below-average viewing activity. "
                    "Consider recommending a lower-cost plan if the customer is not making sufficient "
                    "use of Premium features, which may improve value perception and retention."
                ),
                'priority': 'Medium'
            })

    else:  # Low risk
        recommendations.append({
            'action': '✅ No Immediate Retention Intervention Required',
            'detail': (
                f"This customer's churn probability is {churn_pct:.1f}%, which is classified as low risk. "
                f"Continue normal customer engagement and monitoring. No immediate retention action is needed."
            ),
            'priority': 'Low'
        })

    return recommendations


# ==========================================
# HELPER: GENERATE AI-ASSISTED RETENTION STRATEGY
# ==========================================
def generate_ai_retention_strategy(customer_row, risk_level, churn_pct, stats, api_key):
    """Generate AI-assisted retention strategy using external generative AI service."""
    prompt = f"""You are a Netflix customer retention specialist AI assistant. Analyze the following customer profile and their churn risk assessment, then provide personalized, actionable retention strategies.

**Customer Profile:**
- Customer ID: {customer_row['customer_id']}
- Age: {int(customer_row['age'])} | Gender: {customer_row['gender']} | Region: {customer_row['region']}
- Subscription Plan: {customer_row['subscription_type']} (${customer_row['monthly_fee']:.2f}/month)
- Payment Method: {customer_row['payment_method']}
- Primary Device: {customer_row['device']}
- Favourite Genre: {customer_row['favorite_genre']}
- Number of Profiles: {int(customer_row['number_of_profiles'])}

**Engagement Metrics (Customer vs Dataset Average):**
- Total Watch Hours: {customer_row['watch_hours']:.1f} hrs (Dataset Avg: {stats['avg_watch_hours']:.1f} hrs)
- Avg Daily Watch Time: {customer_row['avg_watch_time_per_day']:.2f} hrs (Dataset Avg: {stats['avg_daily_watch']:.2f} hrs)
- Inactivity Days (Days Since Last Login): {int(customer_row['last_login_days'])} days (Dataset Avg: {stats['avg_inactivity']:.0f} days)
- Number of Profiles: {int(customer_row['number_of_profiles'])} (Dataset Avg: {stats['avg_profiles']:.1f})

**Churn Risk Assessment:**
- Predicted Churn Probability: {churn_pct:.1f}%
- Risk Level: {risk_level}

Please provide:
1. **Risk Analysis**: A concise 2-3 sentence assessment explaining the key factors contributing to this customer's churn risk.
2. **Retention Strategies**: Provide 2-4 specific, actionable retention strategies tailored to this customer's unique profile. For each strategy include:
   - An emoji icon and clear strategy name
   - Detailed action steps
   - Priority level (High / Medium / Low)
   - Expected impact

Keep the response professional, concise, and directly actionable by retention staff. Format using markdown."""

    clean_key = str(api_key).strip()

    # 1. Support Groq AI (Free, ultra-fast LLMs)
    if clean_key.startswith("gsk_"):
        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json"
        }
        
        # Dynamically fetch active model list from Groq
        active_models = []
        try:
            m_resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            if m_resp.status_code == 200:
                data = m_resp.json().get("data", [])
                active_models = [m["id"] for m in data if "whisper" not in m.get("id", "").lower()]
        except Exception:
            pass

        if not active_models:
            active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]

        last_groq_err = None
        for g_model in active_models:
            try:
                payload = {
                    "model": g_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    err_data = response.json().get("error", {})
                    last_groq_err = f"Groq Error ({g_model}): {err_data.get('message', response.text)}"
            except Exception as e:
                last_groq_err = str(e)

        if last_groq_err:
            raise Exception(last_groq_err)

    # 2. Support OpenAI API
    if clean_key.startswith("sk-"):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            err_data = response.json().get("error", {})
            raise Exception(f"OpenAI API Error ({response.status_code}): {err_data.get('message', response.text)}")

    # 3. Google Gemini: Modern SDK
    last_error = None
    try:
        client = genai.Client(api_key=clean_key)
        for m in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-flash']:
            try:
                res = client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return res.text
            except Exception as e:
                last_error = e
                continue
    except Exception as e:
        last_error = e

    # 4. Google Gemini: Direct REST fallback
    for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            headers = {
                "x-goog-api-key": clean_key,
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                err_data = response.json().get("error", {})
                last_error = Exception(f"HTTP {response.status_code}: {err_data.get('message', response.text)}")
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error
    raise Exception("Unknown error communicating with external generative AI service.")


# ==========================================
# MAIN APPLICATION TITLE BANNER
# ==========================================
logo_b64 = get_logo_base64("Netflix_Logo.png")
if logo_b64:
    logo_img_html = f'<img src="data:image/png;base64,{logo_b64}" height="55" style="object-fit: contain;" />'
else:
    logo_img_html = ''

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; padding-bottom: 5px;">
    {logo_img_html}
    <h1 style="color: #B81D24; margin: 0; font-weight: 800; font-size: 2.2rem; line-height: 1.2;">Netflix Churn Prediction & Retention Support System</h1>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR FILTER BAR
# ==========================================
st.sidebar.markdown("## Dataset Filter Bar")
st.sidebar.markdown("<small style='color: #666;'>Filter customer records across chart visualizations</small>", unsafe_allow_html=True)

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

# ---- AI Configuration ----
ai_api_key = None
try:
    if "ai" in st.secrets and "api_key" in st.secrets["ai"]:
        ai_api_key = st.secrets["ai"]["api_key"]
    elif "api_key" in st.secrets:
        ai_api_key = st.secrets["api_key"]
    elif "AI_API_KEY" in st.secrets:
        ai_api_key = st.secrets["AI_API_KEY"]
    elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        ai_api_key = st.secrets["gemini"]["api_key"]
    elif "GEMINI_API_KEY" in st.secrets:
        ai_api_key = st.secrets["GEMINI_API_KEY"]
    elif "groq" in st.secrets and "api_key" in st.secrets["groq"]:
        ai_api_key = st.secrets["groq"]["api_key"]
    elif "GROQ_API_KEY" in st.secrets:
        ai_api_key = st.secrets["GROQ_API_KEY"]
    elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        ai_api_key = st.secrets["openai"]["api_key"]
    elif "OPENAI_API_KEY" in st.secrets:
        ai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    ai_api_key = None

if not ai_api_key:
    for env_var in ["AI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
        if os.environ.get(env_var):
            ai_api_key = os.environ.get(env_var)
            break

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Churn Overview",
    "⚠️ At-Risk Customers",
    "🔍 Customer Assessment & Retention Strategy",
    "📈 Analytics & Exploration",
    "🤖 Model Evaluation",
    "🎯 Individual Churn Predictor"
])


# ==========================================
# TAB 1: CHURN OVERVIEW
# ==========================================
with tab1:
    st.markdown("### Churn Overview Dashboard")
    st.markdown("<small style='color: #666;'>Key performance indicators and churn breakdown summary for operational decision-making</small>", unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("⚠️ No records match the selected sidebar filters. Please broaden your filter criteria.")
    else:
        total_records = len(filtered_df)
        churn_count = int(filtered_df['churned'].sum())
        retained_count = total_records - churn_count
        churn_rate = (churn_count / total_records * 100) if total_records > 0 else 0
        avg_inactivity = filtered_df['last_login_days'].mean()
        avg_watch = filtered_df['watch_hours'].mean()
        
        # KPI Cards
        st.markdown("#### Key Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(label="Total Customers", value=f"{total_records:,}")
        with kpi2:
            st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%", delta=f"{churn_count:,} churned", delta_color="inverse")
        with kpi3:
            st.metric(label="Avg Inactivity Days", value=f"{avg_inactivity:.0f} days")
        with kpi4:
            st.metric(label="Avg Watch Hours", value=f"{avg_watch:.1f} hrs")
        
        st.markdown("---")
        
        # Risk Distribution (always Random Forest)
        st.markdown("#### Customer Risk Distribution")
        st.caption("We use Random Forest as our dafault model because it achieved the best overall performance among the evaluated models.")
        
        risk_counts = scored_df['risk_level'].value_counts()
        high_count = int(risk_counts.get('High', 0))
        med_count = int(risk_counts.get('Medium', 0))
        low_count = int(risk_counts.get('Low', 0))
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            st.markdown(f"""
            <div style="background: #FEF2F2; border-left: 4px solid #B81D24; padding: 15px; border-radius: 6px;">
                <div style="font-size: 0.85rem; color: #991B1B; font-weight: 600;">🔴 HIGH RISK</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #B81D24;">{high_count:,}</div>
                <div style="font-size: 0.75rem; color: #666;">Churn probability ≥ 65%</div>
            </div>
            """, unsafe_allow_html=True)
        with risk_col2:
            st.markdown(f"""
            <div style="background: #FFFBEB; border-left: 4px solid #D97706; padding: 15px; border-radius: 6px;">
                <div style="font-size: 0.85rem; color: #92400E; font-weight: 600;">🟡 MEDIUM RISK</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #D97706;">{med_count:,}</div>
                <div style="font-size: 0.75rem; color: #666;">Churn probability ≥ 35%</div>
            </div>
            """, unsafe_allow_html=True)
        with risk_col3:
            st.markdown(f"""
            <div style="background: #F0FDF4; border-left: 4px solid #16A34A; padding: 15px; border-radius: 6px;">
                <div style="font-size: 0.85rem; color: #166534; font-weight: 600;">🟢 LOW RISK</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #16A34A;">{low_count:,}</div>
                <div style="font-size: 0.75rem; color: #666;">Churn probability < 35%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset Summary & Churn Breakdown Pie
        col_text, col_target_pie = st.columns([1.2, 1])
        with col_text:
            st.markdown("#### Dataset Summary")
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
        
        # Subscription Breakdown Summary
        st.markdown("#### Subscription Breakdown Summary")
        basic_count = len(filtered_df[filtered_df['subscription_type'] == 'Basic'])
        std_count = len(filtered_df[filtered_df['subscription_type'] == 'Standard'])
        prem_count = len(filtered_df[filtered_df['subscription_type'] == 'Premium'])
        
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


# ==========================================
# TAB 2: AT-RISK CUSTOMERS
# ==========================================
with tab2:
    st.markdown("### At-Risk Customers")
    st.markdown("<small style='color: #666;'>This tab is for staff to prioritise customers who may require attention.</small>", unsafe_allow_html=True)

    # ---- Filter Controls ----
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        risk_filter = st.selectbox(
            "Risk Level",
            options=['All', 'High', 'Medium', 'Low'],
            index=0,
            key='atrisk_risk_filter'
        )

    with ctrl_col2:
        sub_filter = st.selectbox(
            "Subscription Type",
            options=['All', 'Basic', 'Standard', 'Premium'],
            index=0,
            key='atrisk_sub_filter'
        )

    with ctrl_col3:
        show_count = st.selectbox(
            "Show",
            options=['Top 10', 'Top 25', 'Top 50', 'All'],
            index=1,
            key='atrisk_show_count'
        )

    # ---- Apply filters ----
    display_df = scored_df.copy()

    if risk_filter != 'All':
        display_df = display_df[display_df['risk_level'] == risk_filter]

    if sub_filter != 'All':
        display_df = display_df[display_df['subscription_type'] == sub_filter]

    # Apply row limit
    show_map = {'Top 10': 10, 'Top 25': 25, 'Top 50': 50, 'All': len(display_df)}
    limit = show_map.get(show_count, 25)
    display_df = display_df.head(limit)

    # ---- Summary ----
    st.markdown(
        f"**Showing {len(display_df):,} customers** | "
        f"🔴 High: {len(display_df[display_df['risk_level'] == 'High']):,} | "
        f"🟡 Medium: {len(display_df[display_df['risk_level'] == 'Medium']):,} | "
        f"🟢 Low: {len(display_df[display_df['risk_level'] == 'Low']):,}"
    )

    # ---- Display Table ----
    table_df = display_df[[
        'customer_id', 'churn_pct', 'risk_level', 'subscription_type',
        'watch_hours', 'avg_watch_time_per_day', 'last_login_days'
    ]].copy()
    table_df.columns = [
        'Customer ID', 'Churn Prob (%)', 'Risk Level', 'Subscription',
        'Watch Hours', 'Avg Daily Watch (hrs)', 'Inactivity Days'
    ]

    st.dataframe(
        table_df,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            'Churn Prob (%)': st.column_config.ProgressColumn(
                "Churn Prob (%)",
                help="Predicted churn probability from Random Forest",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
        }
    )


# ==========================================
# TAB 3: CUSTOMER ASSESSMENT & RETENTION STRATEGY
# ==========================================
with tab3:
    st.markdown("### Customer Assessment & Retention Strategy")
    st.markdown("<small style='color: #666;'>This tab shows the risk rofile, engagement summary and retention recommendations for the selected customer.</small>", unsafe_allow_html=True)


    # ---- Customer Selection ----
    st.markdown("---")
    st.markdown("#### Select a Customer for Assessment")

    if not display_df.empty:
        # Build meaningful labels for the selectbox
        select_options = []
        for _, row in display_df.iterrows():
            label = (
                f"{row['customer_id']} — {row['churn_pct']}% "
                f"{row['risk_level']} Risk — {row['subscription_type']}"
            )
            select_options.append(label)

        selected_label = st.selectbox(
            "Choose a customer:",
            options=select_options,
            index=0,
            key='atrisk_customer_select'
        )

        # Extract customer_id from label
        selected_cid = selected_label.split(" — ")[0]
        st.session_state['selected_customer_id'] = selected_cid

    selected_cid = st.session_state.get('selected_customer_id', None)

    if not selected_cid:
        st.info("👈 Please select a customer from the **⚠️ At-Risk Customers** tab first.")
    else:
        match = scored_df[scored_df['customer_id'] == selected_cid]

        if match.empty:
            st.error(f"❌ Customer ID '{selected_cid}' not found in the dataset.")
        else:
            customer = match.iloc[0]
            churn_pct = float(customer['churn_pct'])
            risk_level = str(customer['risk_level'])

            # ============================================
            # SECTION A — Customer Risk Profile
            # ============================================
            st.markdown("---")
            st.markdown("#### A. Customer Risk Profile")

            profile_col1, profile_col2 = st.columns([1.3, 1])

            with profile_col1:
                # Risk level badge
                if risk_level == 'High':
                    st.error(f"⚠️ **HIGH RISK** — Churn Probability: {churn_pct}%")
                elif risk_level == 'Medium':
                    st.warning(f"⚡ **MEDIUM RISK** — Churn Probability: {churn_pct}%")
                else:
                    st.success(f"✅ **LOW RISK** — Churn Probability: {churn_pct}%")

                # Customer information table
                st.markdown(f"""
| Attribute | Value |
|---|---|
| **Customer ID** | `{customer['customer_id']}` |
| **Age** | {int(customer['age'])} |
| **Gender** | {customer['gender']} |
| **Region** | {customer['region']} |
| **Subscription Plan** | {customer['subscription_type']} (${customer['monthly_fee']:.2f}/month) |
| **Payment Method** | {customer['payment_method']} |
| **Primary Device** | {customer['device']} |
| **Favourite Genre** | {customer['favorite_genre']} |
| **Number of Profiles** | {int(customer['number_of_profiles'])} |
| **Total Watch Hours** | {customer['watch_hours']:.1f} hrs |
| **Avg Daily Watch Time** | {customer['avg_watch_time_per_day']:.2f} hrs |
| **Inactivity Days** | {int(customer['last_login_days'])} days |
                """)

            with profile_col2:
                # Churn Probability Gauge
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
                fig_gauge.update_layout(template="plotly_white", height=200, margin=dict(l=15, r=15, t=35, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ============================================
            # SECTION B — Quick Comparison with Dataset Averages
            # ============================================
            st.markdown("---")
            st.markdown("#### B. Quick Comparison with Dataset Averages")

            compare_data = pd.DataFrame([
                {
                    "Measure": "Watch Hours",
                    "Customer": f"{customer['watch_hours']:.1f}",
                    "Dataset Average": f"{dataset_stats['avg_watch_hours']:.1f}"
                },
                {
                    "Measure": "Avg Daily Watch (hrs)",
                    "Customer": f"{customer['avg_watch_time_per_day']:.2f}",
                    "Dataset Average": f"{dataset_stats['avg_daily_watch']:.2f}"
                },
                {
                    "Measure": "Inactivity Days",
                    "Customer": f"{int(customer['last_login_days'])}",
                    "Dataset Average": f"{dataset_stats['avg_inactivity']:.0f}"
                },
                {
                    "Measure": "Number of Profiles",
                    "Customer": f"{int(customer['number_of_profiles'])}",
                    "Dataset Average": f"{dataset_stats['avg_profiles']:.1f}"
                },
            ])
            st.dataframe(compare_data, use_container_width=True, hide_index=True)

            # ============================================
            # SECTION C — Customer Engagement Summary
            # ============================================
            st.markdown("---")
            st.markdown("#### C. Customer Engagement Summary")

            observations = generate_engagement_summary(customer, dataset_stats)
            for obs in observations:
                st.markdown(f"- {obs}", unsafe_allow_html=True)

            # ============================================
            # SECTION D — AI-Assisted Retention Strategy
            # ============================================
            st.markdown("---")
            st.markdown("#### D. AI-Assisted Retention Strategy")

            if ai_api_key:
                # --- AI-Assisted Recommendations (External Generative AI Service) ---
                st.markdown(
                    '<small style="color: #666;">Powered by external generative AI service</small>',
                    unsafe_allow_html=True
                )

                # Track AI responses per customer in session state
                ai_state_key = f"ai_retention_{selected_cid}"

                if st.button("🤖 Generate AI-Assisted Retention Strategy", key="gen_ai_btn", type="primary"):
                    with st.spinner("🔄 Generating AI-assisted retention strategy..."):
                        try:
                            ai_response = generate_ai_retention_strategy(
                                customer, risk_level, churn_pct, dataset_stats, ai_api_key
                            )
                            st.session_state[ai_state_key] = ai_response
                        except Exception as e:
                            st.error(f"❌ Failed to generate AI recommendation: {e}")

                # Display the AI response if available
                if ai_state_key in st.session_state:
                    st.markdown(
                        f'<div style="background: #F8F9FA; border: 1px solid #E9ECEF; '
                        f'border-left: 4px solid #B81D24; padding: 18px 20px; '
                        f'border-radius: 8px; margin-top: 10px;">',
                        unsafe_allow_html=True
                    )
                    st.markdown(st.session_state[ai_state_key])
                    st.markdown('</div>', unsafe_allow_html=True)

                st.caption(
                    "These recommendations are generated by an external generative AI service based on the customer's "
                    "profile, engagement metrics, and predicted churn risk. They are decision-support "
                    "suggestions and should be reviewed by retention staff before any action is taken."
                )

            else:
                # --- Fallback: Rule-Based Recommendations ---
                st.info(
                    "💡 **AI API key not configured.** Showing rule-based retention recommendations below. "
                    "To enable AI-assisted retention strategies, configure your AI API key in `.streamlit/secrets.toml`."
                )

                recommendations = generate_recommendations(customer, risk_level, churn_pct, dataset_stats)

                for rec in recommendations:
                    priority_colors = {'High': '#FEF2F2', 'Medium': '#FFFBEB', 'Low': '#F0FDF4'}
                    priority_borders = {'High': '#B81D24', 'Medium': '#D97706', 'Low': '#16A34A'}
                    bg_color = priority_colors.get(rec['priority'], '#F3F4F6')
                    border_color = priority_borders.get(rec['priority'], '#999')

                    st.markdown(f"""
<div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
    <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">{rec['action']}</div>
    <div style="font-size: 0.85rem; color: #374151;">{rec['detail']}</div>
    <div style="font-size: 0.72rem; color: #666; margin-top: 6px;">Priority: {rec['priority']}</div>
</div>
                    """, unsafe_allow_html=True)

                st.caption(
                    "These are rule-based decision-support suggestions. Configure your AI API key "
                    "in `.streamlit/secrets.toml` to generate AI-assisted personalized recommendations."
                )


# ==========================================
# TAB 4: ANALYTICS & EXPLORATION
# ==========================================
with tab4:
    st.markdown("### Analytics & Exploration")
    st.markdown("<small style='color: #666;'>Visualising churn risk factors and exploring customer data using filtered records</small>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No records match the current sidebar filter selection. Please adjust your filters.")
    else:
        # ---- VISUALISATION DASHBOARD (all 4 existing charts preserved) ----
        st.markdown("#### Churn Visualisation Dashboard")
        grid_row1_col1, grid_row1_col2 = st.columns(2)

        # GRAPH 1: Subscription Type vs. Churn Rate
        with grid_row1_col1:
            st.markdown("##### 1. Subscription Type vs. Churn Rate")
            g1_data = filtered_df.groupby('subscription_type')['churned'].agg(
                Total='count',
                Churned='sum',
                ChurnRate=lambda x: (x.sum() / x.count()) * 100
            ).reset_index()

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

        # GRAPH 2: Watch Hours Distribution by Churn Status
        with grid_row1_col2:
            st.markdown("##### 2. Watch Hours Distribution by Churn Status")
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

        # GRAPH 3: Number of Profiles vs. Churn Rate
        with grid_row2_col1:
            st.markdown("##### 3. Number of Profiles vs. Churn Rate")
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

        # GRAPH 4: Inactivity Days vs. Churn Rate
        with grid_row2_col2:
            st.markdown("##### 4. Inactivity Days vs. Churn Rate")
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

        # ---- DATASET PREVIEW ----
        st.markdown("---")
        st.markdown("#### Dataset Preview")

        row_options = [5, 10, 20, 50, 100, "All"]
        selected_rows = st.selectbox("Rows to view:", options=row_options, index=2)

        preview_df = filtered_df.drop(columns=['customer_id'], errors='ignore')

        if selected_rows != "All":
            preview_df = preview_df.head(int(selected_rows))

        st.dataframe(preview_df, use_container_width=True, height=350)


# ==========================================
# TAB 5: MODEL EVALUATION
# ==========================================
with tab5:
    st.markdown("### Machine Learning Model Evaluation")
    st.markdown("<small style='color: #666;'>Academic comparison of KNN, Logistic Regression, and Random Forest on held-out test data (20% split)</small>", unsafe_allow_html=True)

    # Metrics Table
    st.markdown("#### Model Evaluation Performance Table")

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
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Actual vs Predicted Performance Visualizations")

    col_cm, col_pred_compare = st.columns([1, 1])

    with col_cm:
        st.markdown("##### Confusion Matrices")
        selected_cm_model = st.selectbox(
            "Select Model for Confusion Matrix:",
            options=['Random Forest', 'Logistic Regression', 'KNN']
        )

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
        pred_comp_data.append({'Model': 'Actual (Ground Truth)', 'Status': 'Retained (0)', 'Count': (y_test == 0).sum()})
        pred_comp_data.append({'Model': 'Actual (Ground Truth)', 'Status': 'Churned (1)', 'Count': (y_test == 1).sum()})

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

    # Conclusion
    st.markdown("---")
    st.markdown("#### Model Selection Conclusion")
    st.info(
        "Random Forest achieved the strongest overall performance among the three evaluated "
        "classification models. Therefore, it is used as the default operational model for "
        "customer risk scoring and retention assessment throughout this system, while KNN and "
        "Logistic Regression are retained for academic comparison."
    )


# ==========================================
# TAB 6: INDIVIDUAL CHURN PREDICTOR
# ==========================================
with tab6:
    st.markdown("### Individual Customer Churn Predictor")
    st.markdown("<small style='color: #666;'>Input custom customer features to simulate and predict churn risk in real-time. This is for testing hypothetical customers who are not in the existing dataset.</small>", unsafe_allow_html=True)

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
        selected_pred_model = st.selectbox(
            "Choose Prediction Algorithm:",
            options=['Random Forest', 'Logistic Regression', 'KNN'],
            index=0
        )
        st.caption("Random Forest is selected by default because it achieved the best overall evaluation performance. Other models are available for comparison.")

    with col_predict_btn:
        st.write("")
        st.write("")
        predict_submitted = st.button("Predict Churn Risk", use_container_width=True, type="primary")

    if predict_submitted:
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
