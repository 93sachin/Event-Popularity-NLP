import streamlit as st
import pandas as pd
import re
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(words)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/events.csv")
    df = df.dropna()
    df["text"] = df["description"].apply(clean_text)
    return df

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(df["text"])

    numeric = df[["price", "past_attendance"]]
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)

    X = hstack([text_features, numeric_scaled])
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, scaler, acc

model, vectorizer, scaler, acc = train_model(df)

# ---------------- UI ----------------
st.title("🎯 Event Popularity Predictor")
st.write("Predict whether your event will be popular or not 🔥")

# Inputs
description = st.text_area("Enter Event Description")
price = st.number_input("Enter Price", min_value=0.0, step=10.0)
attendance = st.number_input("Enter Expected Attendance", min_value=0.0, step=10.0)

# Prediction
if st.button("Predict"):

    clean_input = clean_text(description)
    text_vec = vectorizer.transform([clean_input])

    input_df = pd.DataFrame([[price, attendance]],
                            columns=["price", "past_attendance"])
    num_vec = scaler.transform(input_df)

    final_input = hstack([text_vec, num_vec])

    prediction = model.predict(final_input)[0]

    if prediction == 1:
        st.success("🔥 This event is likely to be POPULAR")
    else:
        st.error("⚠️ This event is likely NOT popular")