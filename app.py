import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Subscription & Churn Portal",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Price Mapping & Session State Setup
# ---------------------------------------------------------
PRICES = {
    "Basic": 8.99,
    "Standard": 13.99,
    "Premium": 17.99
}

# 1. Initialize fee_display directly in session_state
if "fee_display" not in st.session_state:
    st.session_state.fee_display = PRICES["Basic"]

# 2. Callback function updating fee_display directly when dropdown changes
def update_fee():
    selected_plan = st.session_state.subscription_type
    st.session_state.fee_display = PRICES[selected_plan]


# ---------------------------------------------------------
# 3. Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Subscription Form", "View Dataset Metrics"])


# ---------------------------------------------------------
# PAGE 1: Subscription Form
# ---------------------------------------------------------
if page == "Subscription Form":
    st.title("🎬 Netflix Customer Subscription Interface")
    st.write("Register a customer account or update plan details below. The fee defaults automatically to the plan's exact price.")
    
    st.divider()

    st.subheader("1. Customer Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=12, max_value=100, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    
    with col2:
        region = st.selectbox("Region", options=["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
        device = st.selectbox("Primary Device", options=["TV", "Mobile", "Desktop", "Tablet"])

    with col3:
        payment_method = st.selectbox("Payment Method", options=["Gift Card", "Crypto", "Credit Card", "PayPal"])
        number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=5, value=1)

    st.subheader("2. Subscription & Preferences")
    col4, col5 = st.columns(2)

    with col4:
        # Subscription Type Selectbox
        st.selectbox(
            label="Subscription Type",
            options=list(PRICES.keys()),
            key="subscription_type",
            on_change=update_fee,
            help="Selecting a plan automatically sets the monthly fee."
        )
        
        # Monthly Fee display linked directly to st.session_state.fee_display
        st.number_input(
            label="Monthly Fee ($)",
            key="fee_display",
            format="%.2f",
            disabled=True,
            help="Price automatically set based on subscription type."
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
            "Monthly Fee": f"${st.session_state.fee_display:.2f}",
            "Favorite Genre": favorite_genre,
            "Avg Daily Watch Time (hrs)": avg_watch_time
        }
        st.json(submitted_data)


# ---------------------------------------------------------
# PAGE 2: View Dataset Metrics
# ---------------------------------------------------------
elif page == "View Dataset Metrics":
    st.title("📊 Netflix Dataset Overview")
    
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("netflix_customer_churn.csv")
        except Exception:
            return None

    df = load_data()
    
    if df is not None:
        st.write("Overview of the `netflix_customer_churn.csv` dataset:")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Basic Subscribers Fee", "$8.99")
        col3.metric("Churn Rate", f"{(df['churned'].mean() * 100):.1f}%")

        st.subheader("Sample Raw Data")
        st.dataframe(df.head(10))
    else:
        st.warning("`netflix_customer_churn.csv` not found in the root directory. Please make sure it is committed to your GitHub repository.")
