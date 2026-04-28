import streamlit as st
import pandas as pd
import re
import spacy

import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# Load data
df = pd.read_csv("data/events.csv")
df["text"] = df["description"].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(max_features=100)
text_features = vectorizer.fit_transform(df["text"])

numeric_features = df[["price", "past_attendance"]]
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

X = hstack([text_features, numeric_scaled])
y = df["popularity"]


# ================= UI =================

st.title("🎯 Event Popularity Predictor")
st.write("Predict whether your event will be popular or not 🔥")

# Inputs
description = st.text_area("Enter Event Description")
price = st.number_input("Enter Price", min_value=0.0, step=10.0)
attendance = st.number_input("Enter Expected Attendance", min_value=0.0, step=10.0)

# Button
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