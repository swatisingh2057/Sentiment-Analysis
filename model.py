import pandas as pd
import numpy as np
from utils import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_text


def load_data(path):
    df = pd.read_csv(path)

    # Convert labels to numeric
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Apply preprocessing
    df['processed'] = df['review'].apply(preprocess_text)

    return df


def train_models(df):
    X = df['processed']
    y = df['sentiment']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    # 🔥 Training Loop
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n=====", name, "=====")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)

        # Save results
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

        # 🔥 Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name)

    return models, vectorizer, results


def predict(text, model, vectorizer):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]

    return "Positive" if pred == 1 else "Negative"