import pandas as pd
import re
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]

    return " ".join(words)

# Load dataset
df = pd.read_csv("data/events.csv")

print("Total data:", len(df))

# Preprocess text
df["text"] = df["description"].apply(clean_text)

# ===================== FEATURES =====================

# TEXT FEATURES (TF-IDF)
vectorizer = TfidfVectorizer(max_features=100)
text_features = vectorizer.fit_transform(df["text"])

# NUMERIC FEATURES
numeric_features = df[["price", "past_attendance"]]

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

# COMBINE BOTH
X = hstack([text_features, numeric_scaled])

# TARGET
y = df["popularity"]

# ===================== TRAIN TEST =====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ===================== MODEL =====================

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

# ===================== PREDICTION =====================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ===================== LIVE PREDICTION =====================

print("\n=== LIVE EVENT PREDICTION ===")

user_input = input("Enter event description: ")
price_input = float(input("Enter price: "))
attendance_input = float(input("Enter past attendance: "))

clean_input = clean_text(user_input)
text_vec = vectorizer.transform([clean_input])

import pandas as pd

input_df = pd.DataFrame(
    [[price_input, attendance_input]],
    columns=["price", "past_attendance"]
)

num_vec = scaler.transform(input_df)

final_input = hstack([text_vec, num_vec])

prediction = model.predict(final_input)[0]

if prediction == 1:
    print("🔥 This event is likely to be POPULAR")
else:
    print("⚠️ This event is likely NOT popular")