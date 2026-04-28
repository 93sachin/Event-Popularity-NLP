import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("data/events.csv")

# Use cleaned text (IMPORTANT)
# Agar save nahi kiya tha to temporary cleaning kar le
df["text"] = df["description"].str.lower()

# TF-IDF
vectorizer = TfidfVectorizer(max_features=100)

X = vectorizer.fit_transform(df["text"])

# Convert to dataframe
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(tfidf_df.head())