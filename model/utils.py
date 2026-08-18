import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

NUMERICAL_FEATURES = [
    'age', 'watch_hours', 'last_login_days', 'monthly_fee', 
    'number_of_profiles', 'avg_watch_time_per_day'
]

CATEGORICAL_FEATURES = [
    'gender', 'subscription_type', 'region', 'device', 
    'payment_method', 'favorite_genre'
]

TARGET = 'churned'

def load_data(filepath='netflix_customer_churn.csv'):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def get_preprocessor():
    """Build ColumnTransformer for preprocessing numerical and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_true, y_pred):
    """Calculate accuracy, precision, recall, and f1 score."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1-Score': float(f1),
        'Confusion Matrix': cm
    }