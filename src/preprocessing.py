import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords

# Load model
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words("english"))

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    doc = nlp(" ".join(words))
    words = [token.lemma_ for token in doc]

    return " ".join(words)

# Load data
df = pd.read_csv("data/events.csv")

# Apply cleaning
df["cleaned_description"] = df["description"].apply(clean_text)

# Output check
print(df[["description", "cleaned_description"]].head())