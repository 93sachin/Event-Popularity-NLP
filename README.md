# 🎯 Event Popularity Prediction (NLP + ML)

Predict whether an event will be popular using its **description, price, and past attendance**.

## 🔧 Tech Stack
- Python, Pandas, NumPy  
- spaCy (text preprocessing)  
- Scikit-learn (TF-IDF, Logistic Regression)  
- Streamlit (web app)

## 🧠 What it does
- Cleans text (lowercase, remove noise, lemmatization)
- Converts text to vectors (TF-IDF)
- Combines **text + numeric features (price, attendance)**
- Trains Logistic Regression with class balancing
- Provides **real-time prediction UI** via Streamlit

## 📊 Model Performance
- Accuracy: ~89%
- Balanced performance (see confusion matrix & report in `src/model.py`)

## 🖥️ App Preview
![App Screenshot](assets/app.png)

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py