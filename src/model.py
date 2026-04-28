import pandas as pd
import re
import spacy
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]

    return " ".join(words)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/events.csv")
df = df.dropna()

print("Total data:", len(df))

# ---------------- TEXT ----------------
df["text"] = df["description"].astype(str)
df["text"] = df["text"].apply(clean_text)

# ---------------- FEATURES ----------------

# TEXT
vectorizer = TfidfVectorizer(max_features=200)
text_features = vectorizer.fit_transform(df["text"])

# NUMERIC
numeric = df[["price", "past_attendance"]]

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric)

# COMBINE
X = hstack([text_features, numeric_scaled])

# TARGET
y = df["popularity"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# ---------------- SAVE ----------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------- LIVE ----------------
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