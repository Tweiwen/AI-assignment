from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from model.utils import get_preprocessor, evaluate_predictions

def train_rf_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier wrapped in a preprocessing pipeline.
    """
    preprocessor = get_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_rf(model, X):
    """
    Predict churn labels and probabilities using Random Forest model.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_proba

def evaluate_rf(model, X_test, y_test):
    """
    Evaluate Random Forest model performance on test dataset.
    """
    y_pred, _ = predict_rf(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return metrics