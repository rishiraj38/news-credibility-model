import joblib
import numpy as np
from newspaper import Article
from .preprocess import clean_text


model_path = 'best_model.pkl'


def load_model():
    """Load trained pipeline."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print("Model not found. Train first.")
        return None


def extract_text_from_url(url):
    """Robust URL extraction with fallback."""
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=10)
        html = response.text

        article = Article(url)
        article.set_html(html)
        article.parse()

        text = article.title + " " + article.text

        if len(text) < 100:
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])

        return text
    except Exception:
        return None


def extract_top_phrases(text, vectorizer):
    """Extract top TF-IDF words."""
    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return [feature_names[i] for i in sorted_indices[:5]]


def predict_article(input_text, is_url=False):
    """Predict credibility."""
    model = load_model()
    if model is None:
        return "Model not found", 0.0, []

    if is_url:
        text = extract_text_from_url(input_text)
        if not text:
            return "Failed to extract text", 0.0, []
    else:
        text = input_text

    cleaned_text = clean_text(text)

    if len(cleaned_text.split()) < 10:
        return "Article too short", 0.0, []

    # Prediction
    prediction = model.predict([cleaned_text])[0]

    try:
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = max(probabilities) * 100
    except:
        confidence = 0.0

    # Extract top patterns
    vectorizer = model.named_steps["tfidf"]
    top_patterns = extract_top_phrases(cleaned_text, vectorizer)

    label = "High Credibility" if prediction == 1 else "Low Credibility"

    return label, confidence, top_patterns