"""Train/load a compact, local sentiment baseline. No model download required."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_reviews.csv"
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

class SentimentModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def predict_sentiment(self, text: str):
        pred = str(self.pipeline.predict([text])[0])
        probs = self.pipeline.predict_proba([text])[0]
        return pred, float(max(probs))


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=30000, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=30000, strip_accents="unicode")),
    ])
    return Pipeline([("features", features), ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))])


def train_and_evaluate(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path).dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[(df.review != "") & df.sentiment.isin(["Positive", "Negative", "Neutral"])]
    if df.sentiment.nunique() < 3 or df.sentiment.value_counts().min() < 2:
        raise ValueError("Training data must contain at least two examples for each Positive, Negative, and Neutral class.")
    X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment, test_size=0.2, random_state=42, stratify=df.sentiment)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {"test_rows": int(len(X_test)), "accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, average="macro")), "classification_report": report, "note": "Metrics are measured on a held-out split of the bundled synthetic starter dataset; they are not evidence of real-world accuracy."}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return metrics


def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        return None
    return SentimentModel(joblib.load(model_path))
