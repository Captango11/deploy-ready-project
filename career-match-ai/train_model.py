"""Train CareerMatch AI: NLP preprocessing -> TF-IDF -> calibrated Logistic Regression.
Usage: python train_model.py
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.preprocessing import preprocess

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "resumes.csv")
MODEL_DIR = os.path.join(BASE, "model")
RANDOM_STATE = 42


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if not {"resume_text", "job_role"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'resume_text' and 'job_role' columns")
    df = df.dropna(subset=["resume_text", "job_role"])
    df = df[df["resume_text"].astype(str).str.strip().str.len() > 0]
    return df


def get_split(df):
    X = df["resume_text"].astype(str).map(preprocess)
    y = df["job_role"].astype(str)
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def train():
    df = load_data()
    print(f"Loaded {len(df)} resumes, {df['job_role'].nunique()} roles")
    X_train, X_test, y_train, y_test = get_split(df)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                 max_features=10000, min_df=1)
    Xtr = vectorizer.fit_transform(X_train)

    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, y_train)

    # Uncalibrated twin, used ONLY for coefficient-based explanations
    explain_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    metadata = {
        "classes": list(clf.classes_),
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "explain_coef": explain_lr.coef_,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(len(X_test)),
        "n_features": int(Xtr.shape[1]),
        "random_state": RANDOM_STATE,
        "vectorizer_params": {"ngram_range": "(1,2)", "sublinear_tf": True,
                              "max_features": 10000, "min_df": 1},
        "model": "CalibratedClassifierCV(LogisticRegression, sigmoid, cv=3)",
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"Saved model to {MODEL_DIR} (features={Xtr.shape[1]})")
    return clf, vectorizer, metadata


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
