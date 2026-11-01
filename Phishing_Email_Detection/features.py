import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract simple URL and language indicators from each email."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        suspicious_words = [
            "urgent", "verify", "password", "suspended", "locked",
            "winner", "claim", "prize", "refund", "immediately",
            "confirm", "click", "limited", "expires", "payment"
        ]
        rows = []
        for item in X:
            text = str(item)
            lower = text.lower()
            urls = re.findall(r'https?://\S+|www\.\S+', lower)
            has_ip = int(bool(re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', lower)))
            has_shortener = int(any(x in lower for x in ["bit.ly", "tinyurl", "t.co"]))
            https_count = sum(1 for u in urls if u.startswith("https://"))
            http_count = sum(1 for u in urls if u.startswith("http://"))
            keyword_count = sum(lower.count(word) for word in suspicious_words)
            exclamations = text.count("!")
            upper_letters = sum(1 for c in text if c.isupper())
            letters = sum(1 for c in text if c.isalpha())
            upper_ratio = upper_letters / max(letters, 1)
            rows.append([
                len(text), len(urls), has_ip, has_shortener,
                https_count, http_count, keyword_count,
                exclamations, upper_ratio
            ])
        return csr_matrix(np.asarray(rows, dtype=float))
