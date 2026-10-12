"""
Fake News Detector
==================
TF-IDF + Logistic Regression text classifier.

Dataset (recommended): "Fake and Real News Dataset" on Kaggle
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
It contains two files: Fake.csv and True.csv (columns: title, text, subject, date).

Install:
    pip install pandas scikit-learn joblib

Usage:
    # 1) Train (saves model.joblib)
    python fake_news_detector.py train --fake Fake.csv --true True.csv

    # 2) Predict from a string
    python fake_news_detector.py predict --text "Scientists confirm the moon is made of cheese"

    # 3) Predict from a text file
    python fake_news_detector.py predict --file article.txt

    # 4) Interactive mode
    python fake_news_detector.py predict
"""

import argparse
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = "model.joblib"
LABELS = {0: "FAKE", 1: "REAL"}


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, publisher tags, punctuation, extra spaces."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove source tags like "washington (reuters) -" that leak the label in this dataset
    text = re.sub(r"^.{0,60}\(reuters\)\s*-?\s*", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)

    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    df = df[df["content"].str.len() > 0]
    df = df.drop_duplicates(subset="content")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_df=0.7,
                    min_df=5,
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=3.0)),
        ]
    )


def train(args):
    print("Loading data...")
    df = load_data(args.fake, args.true)
    print(f"Samples: {len(df)}  (fake={int((df.label == 0).sum())}, real={int((df.label == 1).sum())})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Training model...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nConfusion matrix (rows=true, cols=pred) [FAKE, REAL]:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    # Show the words that push predictions most strongly toward each class
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    print("Top 'FAKE' indicators:", ", ".join(names[order[:15]]))
    print("Top 'REAL' indicators:", ", ".join(names[order[-15:]]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def predict_one(model: Pipeline, text: str) -> None:
    cleaned = clean_text(text)
    if len(cleaned.split()) < 5:
        print("Text is too short to classify reliably. Paste a headline plus some body text.")
        return
    proba = model.predict_proba([cleaned])[0]
    label = int(proba.argmax())
    print(f"\nPrediction : {LABELS[label]}")
    print(f"Confidence : {proba[label] * 100:.1f}%  (fake={proba[0] * 100:.1f}%, real={proba[1] * 100:.1f}%)")


def predict(args):
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit(f"{MODEL_PATH} not found. Run the 'train' command first.")

    if args.text:
        predict_one(model, args.text)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            predict_one(model, f.read())
    else:
        print("Interactive mode. Paste an article or headline, then press Enter. Type 'q' to quit.")
        while True:
            text = input("\n> ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break
            if text:
                predict_one(model, text)


def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--fake", default="Fake.csv", help="Path to Fake.csv")
    p_train.add_argument("--true", default="True.csv", help="Path to True.csv")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="Classify an article")
    p_pred.add_argument("--text", help="Article text to classify")
    p_pred.add_argument("--file", help="Path to a text file to classify")
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
