# Netflix Customer Churn Prediction System
This project is a supervised machine learning application developed to predict Netflix customer churn and support customer retention decisions.

## Machine Learning Models
Three classification algorithms are implemented:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
The models are evaluated using Accuracy, Precision, Recall, F1-score and Confusion Matrix. Random Forest is selected as the main model for customer risk scoring based on its overall performance.

## Application Features
The Streamlit application provides:
- Churn Overview Dashboard
- At-Risk Customer Identification
- Customer Assessment
- AI-Assisted Retention Strategy
- Analytics and Data Exploration
- Model Performance Comparison
- Individual Customer Churn Prediction

If the external AI service is unavailable, rule-based recommendations are used as a fallback.

### API Configuration
The `secrets.toml` file is private and is not included directly in the project repository. A copy has been provided separately in the submitted ZIP file. Please manually place the provided `secrets.toml` file inside the `.streamlit` folder before running the application.
Expected structure:
.streamlit/
└── secrets.toml

## How to Run
Install the required packages:
pip install -r requirements.txt
Add folder .streamlit and file secrets.toml
Run the application:
streamlit run app.py

## Dataset
The project uses the `netflix_customer_churn.csv` dataset containing customer demographic, subscription and engagement information.

## Purpose
The system demonstrates how machine learning can be used not only to predict customer churn but also to identify high-risk customers and support retention decision-making.