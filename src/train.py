import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .preprocess import preprocess_data


def train_and_evaluate():
    print("Starting preprocessing...")
    df = preprocess_data()

    if df is None:
        print("Dataset not found.")
        return

    X_text = df['content'].values
    y = df['label'].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")

    models = {
        'Logistic_Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=5)),
            ('clf', LogisticRegression(C=0.1, solver="liblinear", max_iter=1000))
        ]),
        'Decision_Tree': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=5)),
            ('clf', DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42))
        ])
    }

    metrics = {}

    # Train & evaluate both models
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1_Score': f1_score(y_test, y_pred)
        }

        print(f"{name} F1: {metrics[name]['F1_Score']:.4f}")

    # ⭐ Force Logistic Regression as final deployed model
    best_model_name = "Logistic_Regression"
    best_pipeline = models[best_model_name]

    print(f"\nFinal Selected Model (Generalization Priority): {best_model_name}")

    metrics['Best_Model'] = best_model_name

    # Save model
    joblib.dump(best_pipeline, 'best_model.pkl')

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Model saved to best_model.pkl")
    print("Metrics saved to metrics.json")


if __name__ == '__main__':
    train_and_evaluate()