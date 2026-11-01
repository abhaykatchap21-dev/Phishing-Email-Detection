from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from features import EmailFeatureExtractor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "emails.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading email dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["email_text", "label"])
X = df["email_text"].astype(str)
y = df["label"].astype(str).str.lower()

if set(y.unique()) != {"safe", "phishing"}:
    raise ValueError("Dataset labels must be exactly 'safe' and 'phishing'.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("features", FeatureUnion([
        ("text_tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=5000
        )),
        ("email_indicators", EmailFeatureExtractor())
    ])),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

print("Training model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=["phishing", "safe"])

joblib.dump(model, MODEL_DIR / "phishing_model.joblib")
metrics = {
    "accuracy": round(float(accuracy), 4),
    "test_samples": int(len(y_test)),
    "classification_report": classification_report(
        y_test, predictions, labels=["phishing", "safe"], output_dict=True, zero_division=0
    ),
    "confusion_matrix": cm.tolist(),
    "labels": ["phishing", "safe"]
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing", "Safe"])
display.plot(cmap="Blues", values_format="d")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
plt.close()

print(f"Model trained successfully. Test accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions, zero_division=0))
print(f"Saved model to: {MODEL_DIR / 'phishing_model.joblib'}")
