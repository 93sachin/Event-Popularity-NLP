import streamlit as st
import pandas as pd
import re
import spacy
import joblib
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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, vectorizer, scaler

model, vectorizer, scaler = load_model()

# ---------------- UI ----------------
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