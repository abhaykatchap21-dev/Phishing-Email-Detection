from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
CHART_PATH = ROOT / "models" / "confusion_matrix.png"

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Phishing Email Detection System")
st.write("An educational machine-learning application that classifies email text as **Phishing** or **Safe**.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The model has not been trained yet. Open the VS Code terminal and run: `python train_model.py`")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

left, right = st.columns([2, 1])
with left:
    st.subheader("Analyze an email")
    email_text = st.text_area(
        "Paste the email message here:",
        height=230,
        placeholder="Example: Your account has been suspended. Verify your password immediately..."
    )
    detect = st.button("🔍 Detect Phishing", type="primary", use_container_width=True)

    if detect:
        if len(email_text.strip()) < 10:
            st.error("Please enter a longer email message to analyze.")
        else:
            prediction = model.predict([email_text])[0]
            probabilities = model.predict_proba([email_text])[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)]) * 100

            if prediction == "phishing":
                st.error("⚠️ Result: PHISHING EMAIL")
                st.write("This message contains patterns commonly associated with phishing.")
            else:
                st.success("✅ Result: LIKELY SAFE EMAIL")
                st.write("The model did not identify strong phishing patterns in this message.")

            st.metric("Model confidence", f"{confidence:.1f}%")
            st.caption("Confidence is the model's estimated probability, not a guarantee that the email is safe.")

with right:
    st.subheader("Model performance")
    st.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    st.metric("Test emails", metrics["test_samples"])
    st.info("Accuracy is measured on the held-out test portion of the included sample dataset.")

st.divider()
st.subheader("Confusion matrix")
if CHART_PATH.exists():
    st.image(str(CHART_PATH), caption="Test-set confusion matrix", use_container_width=True)

with st.expander("Try sample emails"):
    st.markdown("**Phishing example**")
    st.code("URGENT: Your bank account has been suspended. Verify your password immediately at http://secure-account-check.example.com/login")
    st.markdown("**Safe example**")
    st.code("Hi, our project meeting is scheduled for Friday at 10:30 AM. Please review the agenda before we meet.")

st.caption("Educational demonstration only. Do not use this model as the sole basis for security decisions.")
