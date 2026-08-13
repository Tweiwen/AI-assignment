import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report
)

# ---------------------------------------------------------
# 1. DATA LOADING & EXPLORATION
# ---------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

# Load Netflix Churn Dataset
df = pd.read_csv('netflix_customer_churn.csv')

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 records:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING & DATA SPLITTING")
print("=" * 60)

# Separate features and target
X = df.drop(columns=['customer_id', 'churned'])
y = df['churned']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical Features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

# Create Preprocessing Pipeline: Standard Scaling for numbers, One-Hot Encoding for categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Train-Test Split (80% Train, 20% Test) with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size : {X_test.shape[0]} samples")

# ---------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION (KNN, SVM, Random Forest)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TRAINING KNN, SVM, AND RANDOM FOREST")
print("=" * 60)

# Define models required by team members
models = {
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# Train and evaluate each algorithm
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })
    
    print(f"\n--- Detailed Classification Report: {name} ---")
    print(classification_report(y_test, y_pred, digits=4))

# ---------------------------------------------------------
# 4. COMPARATIVE RESULTS SUMMARY
# ---------------------------------------------------------
print("=" * 60)
print("STEP 4: MODEL COMPARISON SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))
