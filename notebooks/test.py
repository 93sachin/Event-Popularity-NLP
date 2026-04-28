import pandas as pd

# Load dataset
df = pd.read_csv("data/events.csv")

print("DATA LOADED SUCCESSFULLY\n")

print(df.head())
print("\nColumns:", df.columns)