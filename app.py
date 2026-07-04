import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# NLTK 
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Model & Vectorizer
@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl",      "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Text Cleaning 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"'\s|'\B", " ", text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(words)

# Streamlit UI 
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛒 Fake Product Review Detection System")
st.write("Detect whether a review is **Fake** or **Genuine** using Machine Learning")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Manual Prediction", "CSV Prediction", "Model Analysis"],
)

# Manual Prediction
if menu == "Manual Prediction":
    st.header("🔍 Predict a Single Review")

    user_review = st.text_area("Enter Review Text")

    if st.button("Predict"):
        if user_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(user_review)
            vector  = vectorizer.transform([cleaned])
            pred    = model.predict(vector)[0]
            proba   = model.predict_proba(vector)[0]

            fake_prob = round(proba[1] * 100, 2)
            gen_prob  = round(proba[0] * 100, 2)

            if pred == 1:
                st.error(f"❌ Fake Review  (Confidence: {fake_prob}%)")
            else:
                st.success(f"✅ Genuine Review  (Confidence: {gen_prob}%)")

            # Explainable AI (XAI) 
            st.subheader("🔍 Explanation")

            feature_names = vectorizer.get_feature_names_out()
            coefficients  = model.coef_[0]
            dense         = vector.toarray()[0]

            important = [
                (feature_names[i], coefficients[i])
                for i in range(len(dense))
                if dense[i] != 0
            ]
            important.sort(key=lambda x: abs(x[1]), reverse=True)
            top_words = [w for w, _ in important[:3]]

            if pred == 1:
                reason = (
                    f"This review is classified as **FAKE** because it contains "
                    f"spam-like or overly promotional words such as: "
                    f"{', '.join(top_words)}."
                )
            else:
                reason = (
                    f"This review is classified as **GENUINE** because it includes "
                    f"natural and experience-based terms such as: "
                    f"{', '.join(top_words)}."
                )

            st.info(reason)


# CSV Prediction
if menu == "CSV Prediction":
    st.header("📂 Upload CSV File")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column!")
        else:
            st.write("Preview:")
            st.dataframe(df.head())

            if st.button("Analyze"):
                vectors = vectorizer.transform(df["review"].apply(clean_text))
                preds   = model.predict(vectors)
                probas  = model.predict_proba(vectors)

                df["Prediction"] = ["Fake" if p == 1 else "Genuine" for p in preds]

                df["Confidence (%)"] = [
                    round(probas[i][1] * 100, 2) if preds[i] == 1
                    else round(probas[i][0] * 100, 2)
                    for i in range(len(preds))
                ]

                st.write("### Results")
                st.dataframe(df[["review", "Prediction", "Confidence (%)"]])

                fake = (df["Prediction"] == "Fake").sum()
                gen  = (df["Prediction"] == "Genuine").sum()

                st.write("### Summary")
                st.write(f"❌ Fake Reviews:    {fake}")
                st.write(f"✅ Genuine Reviews: {gen}")

                fig, ax = plt.subplots()
                ax.bar(["Fake", "Genuine"], [fake, gen], color=["#e05252", "#52a9e0"])
                ax.set_ylabel("Count")
                ax.set_title("Fake vs Genuine Reviews")
                st.pyplot(fig)

# Model Analysis
if menu == "Model Analysis":
    st.header("📊 Model Performance & Visualization")

    if not os.path.exists("test_split.csv"):
        st.error("test_split.csv not found. Re-run train.py to generate it.")
        st.stop()

    df_test = pd.read_csv("test_split.csv")
    X = vectorizer.transform(df_test["clean_review"])
    y = df_test["label"]

    preds    = model.predict(X)
    accuracy = round((preds == y).mean() * 100, 2)
    st.subheader(f"✔ Model Accuracy: {accuracy}%")

    st.subheader("📘 Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
    roc_auc     = round(auc(fpr, tpr), 4)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve (AUC = {roc_auc})")
    st.pyplot(fig_roc)

    st.subheader("📑 Classification Report")
    report = classification_report(y, preds)
    st.text(report)