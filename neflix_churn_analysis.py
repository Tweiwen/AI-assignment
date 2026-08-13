import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_model_comparison():
    df = pd.read_csv('03_Dataset/netflix_customer_churn.csv')
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
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
        trained_pipelines[name] = pipeline

    res_df = pd.DataFrame(results)
    print("=== MODEL EVALUATION SUMMARY ===")
    print(res_df.to_string(index=False))

    joblib.dump(trained_pipelines['Random Forest'], '04_Trained_Model/netflix_churn_pipeline.pkl')
    print("\nBest model (Random Forest) saved to 04_Trained_Model/netflix_churn_pipeline.pkl")

if __name__ == "__main__":
    run_model_comparison()
