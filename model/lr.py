from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from model.utils import get_preprocessor, evaluate_predictions

def train_lr_model(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """
    Train a Logistic Regression classifier wrapped in a preprocessing pipeline.
    """
    preprocessor = get_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=C, max_iter=max_iter, random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_lr(model, X):
    """
    Predict churn labels and probabilities using Logistic Regression model.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_proba

def evaluate_lr(model, X_test, y_test):
    """
    Evaluate Logistic Regression model performance on test dataset.
    """
    y_pred, _ = predict_lr(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return metrics
