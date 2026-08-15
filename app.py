import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Subscription & Analytics Portal",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Standard US Price Mapping & Session State Setup
# ---------------------------------------------------------
PRICES = {
    "Basic": 9.99,
    "Standard": 15.49,
    "Premium": 22.99
}

# Initialize fee_display directly in session_state with US Basic Price ($9.99)
if "fee_display" not in st.session_state:
    st.session_state.fee_display = PRICES["Basic"]

# Callback function updating fee_display directly when dropdown changes
def update_fee():
    selected_plan = st.session_state.subscription_type
    st.session_state.fee_display = PRICES[selected_plan]


# Load Dataset Helper
@st.cache_data
def load_data():
    try:
        return pd.read_csv("netflix_customer_churn.csv")
    except Exception:
        return None

df = load_data()

# ---------------------------------------------------------
# 3. Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Subscription Form", 
    "Subscription Type Analysis", 
    "View Dataset Overview"
])


# ---------------------------------------------------------
# PAGE 1: Subscription Form
# ---------------------------------------------------------
if page == "Subscription Form":
    st.title("🎬 Netflix Customer Registration")
    st.write("Register a customer account or update plan details below. The fee defaults automatically to the plan's exact US price.")
    
    st.divider()

    st.subheader("1. Customer Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=12, max_value=100, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    
    with col2:
        region = st.selectbox("Region", options=["North America", "Europe", "Asia", "Africa", "Oceania", "South America"])
        device = st.selectbox("Primary Device", options=["TV", "Mobile", "Desktop", "Tablet"])

    with col3:
        payment_method = st.selectbox(
            "Payment Method", 
            options=["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"]
        )
        number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=5, value=1)

    st.subheader("2. Subscription & Preferences")
    col4, col5 = st.columns(2)

    with col4:
        # Subscription Type Selectbox (Defaults to Basic)
        st.selectbox(
            label="Subscription Type",
            options=list(PRICES.keys()),
            key="subscription_type",
            on_change=update_fee,
            help="Selecting a plan automatically updates the US monthly fee field."
        )
        
        # Read-only Fee Field synced to selected plan
        st.number_input(
            label="Monthly Fee ($ USD)",
            key="fee_display",
            format="%.2f",
            disabled=True,
            help="US Price automatically set based on subscription type."
        )

    with col5:
        favorite_genre = st.selectbox("Favorite Genre", options=["Action", "Sci-Fi", "Drama", "Horror", "Comedy", "Documentary"])
        avg_watch_time = st.number_input("Avg Watch Time Per Day (Hours)", min_value=0.0, max_value=24.0, value=1.5, step=0.1)

    st.divider()
    
    submit_button = st.button(label="Submit Subscription", type="primary")

    if submit_button:
        st.success("Subscription entry recorded successfully!")
        
        st.subheader("Submitted Record Summary")
        submitted_data = {
            "Age": age,
            "Gender": gender,
            "Region": region,
            "Device": device,
            "Payment Method": payment_method,
            "Number of Profiles": number_of_profiles,
            "Subscription Type": st.session_state.subscription_type,
            "Monthly Fee": f"${st.session_state.fee_display:.2f} USD",
            "Favorite Genre": favorite_genre,
            "Avg Daily Watch Time (hrs)": avg_watch_time
        }
        st.json(submitted_data)


# ---------------------------------------------------------
# PAGE 2: Subscription Type Analysis
# ---------------------------------------------------------
elif page == "Subscription Type Analysis":
    st.title("📈 Subscription Type Analysis")
    
    if df is not None:
        # Group data by subscription type
        sub_stats = df.groupby("subscription_type").agg(
            Total_Customers=("customer_id", "count"),
            Dataset_Avg_Fee=("monthly_fee", "mean"),
            Churn_Rate_Pct=("churned", lambda x: (x.mean() * 100)),
            Avg_Watch_Hours=("watch_hours", "mean"),
            Avg_Daily_Watch_Time=("avg_watch_time_per_day", "mean")
        ).reset_index()

        # Add US Price Column for comparison
        sub_stats["US_Price_USD"] = sub_stats["subscription_type"].map(PRICES)

        st.subheader("Summary Table by Subscription Tier")
        st.dataframe(sub_stats.style.format({
            "US_Price_USD": "${:.2f}",
            "Dataset_Avg_Fee": "${:.2f}",
            "Churn_Rate_Pct": "{:.1f}%",
            "Avg_Watch_Hours": "{:.2f} hrs",
            "Avg_Daily_Watch_Time": "{:.2f} hrs"
        }), use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Churn Rate by Subscription Type")
            fig_churn = px.bar(
                sub_stats, 
                x="subscription_type", 
                y="Churn_Rate_Pct", 
                text_auto=".1f",
                labels={"subscription_type": "Subscription Plan", "Churn_Rate_Pct": "Churn Rate (%)"},
                color="subscription_type"
            )
            st.plotly_chart(fig_churn, use_container_width=True)

        with col_b:
            st.subheader("Customer Distribution by Plan")
            fig_pie = px.pie(
                sub_stats, 
                names="subscription_type", 
                values="Total_Customers",
                color="subscription_type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("Please make sure `netflix_customer_churn.csv` is uploaded in your repository.")


# ---------------------------------------------------------
# PAGE 3: View Dataset Overview
# ---------------------------------------------------------
elif page == "View Dataset Overview":
    st.title("📊 Dataset Metrics")
    
    if df is not None:
        st.write("Overview of `netflix_customer_churn.csv`:")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Basic US Standard Price", f"${PRICES['Basic']:.2f}")
        c3.metric("Overall Churn Rate", f"{(df['churned'].mean() * 100):.1f}%")

        st.subheader("Sample Rows")
        st.dataframe(df.head(15))
    else:
        st.warning("`netflix_customer_churn.csv` not found.")
