# Phishing Email Detection Model

A beginner-friendly educational project using Python, Scikit-learn and Streamlit.

## Requirements
- Python 3.10 or newer (Python 3.11/3.12 recommended)
- VS Code
- VS Code Python extension

## Setup in VS Code

Open this project folder in VS Code. Open Terminal > New Terminal and run:

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will show a local address, usually http://localhost:8501. Open it in your browser.

## Dataset format
`dataset/emails.csv` must contain two columns:
- `email_text`: the email content
- `label`: `phishing` or `safe`

You can replace the included small demonstration dataset with a larger, properly licensed and verified dataset. Keep the same column names and label values, then rerun `python train_model.py`.

## How it works
1. TF-IDF converts email words and phrases into numerical features.
2. EmailFeatureExtractor calculates simple indicators such as URL count, IP-address URLs, URL shorteners, suspicious keyword count and uppercase ratio.
3. Logistic Regression learns patterns from labelled examples.
4. The trained pipeline is saved in `models/phishing_model.joblib`.
5. Streamlit loads the saved model and displays predictions.

## Important limitation
The included 100-row dataset is synthetic and intended only to demonstrate the complete workflow. Its test accuracy is not evidence of real-world phishing detection performance. For a meaningful academic evaluation, use a larger, diverse, labelled dataset, remove duplicates, avoid train/test leakage, and report precision, recall, F1-score and the confusion matrix. A message predicted as safe may still be malicious. Never enter real passwords or confidential emails into this demo.

Project links
Github Repository: https://github.com/abhaykatchap21-dev/Phishing-Email-Detection
Live Demo: https://phishing-email-detection-yitqvkj5uxafn6deanzshc.streamlit.app
