from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from model.utils import get_preprocessor, evaluate_predictions

def train_knn_model(X_train, y_train, n_neighbors=7):
    """
    Train a K-Nearest Neighbors classifier wrapped in a preprocessing pipeline.
    """
    preprocessor = get_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance'))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_knn(model, X):
    """
    Predict churn labels and probabilities using KNN model.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_proba

def evaluate_knn(model, X_test, y_test):
    """
    Evaluate KNN model performance on test dataset.
    """
    y_pred, _ = predict_knn(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return metrics