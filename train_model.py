import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

nltk.download("stopwords", quiet=True)

print("Loading dataset...")
df = pd.read_csv("amazon_reviews.csv")

df["label"] = df["label"].astype(int)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", " ", text)

    words = [
        ps.stem(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

print("\nCleaning reviews...")

# Using review
df["clean_review"] = df["review"].apply(clean_text)

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=2,
    max_df=0.85
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=200,
    C=0.1,
    solver="liblinear"
)

print("\nTraining model...")
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\nModel Accuracy: {acc}%")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save test split
test_split = pd.DataFrame({"clean_review": X_test, "label": y_test})
test_split.to_csv("test_split.csv", index=False)

print("\nFiles saved: model.pkl | vectorizer.pkl | test_split.csv")