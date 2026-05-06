"""
train.py  —  Train the depression detection model
Run this ONCE before starting the Flask app.
"""
import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ── Load dataset ────────────────────────────────────────
CSV_PATH = r"C:/Users/acer/OneDrive/Desktop/depression_dataset_reddit_cleaned.csv"
print(f"Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print("Columns found:", df.columns.tolist())
print("Shape:", df.shape)

# ── Clean text ───────────────────────────────────────────
df['clean_text'] = df['clean_text'].apply(clean_text)

# ── Balance classes ──────────────────────────────────────
majority = df[df['is_depression'] == 0]
minority = df[df['is_depression'] == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_up])
print("Balanced class distribution:\n", df_balanced['is_depression'].value_counts())

# ── Split ────────────────────────────────────────────────
X = df_balanced['clean_text']
y = df_balanced['is_depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Vectorize ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_v = vectorizer.fit_transform(X_train)
X_test_v  = vectorizer.transform(X_test)

# ── Train ─────────────────────────────────────────────────
model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
model.fit(X_train_v, y_train)

# ── Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test_v)
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
pickle.dump(model,      open("model/model.pkl",      "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("\n✅ model/model.pkl and model/vectorizer.pkl saved!")