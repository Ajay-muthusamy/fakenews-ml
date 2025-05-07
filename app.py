from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# Preprocessing function with fixed regex escaping
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Escaped the '[' and ']' properly
    text = re.sub(r"\\W", " ", text)    # Escaped the '\' for non-word characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Escaped the '\S'
    text = re.sub(r'<.*?>+', '', text)   # Removing HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Punctuation
    text = re.sub(r'\n', '', text)       # New lines
    text = re.sub(r'\w*\d\w*', '', text) # Remove digits
    return text

# Output label function
def output_label(n):
    return "Fake News" if n == 0 else "True News"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form["news"]
        data = [wordopt(news)]
        vect = vectorizer.transform(data)
        pred = model.predict(vect)
        result = output_label(pred[0])
        return render_template("index.html", prediction=result, input_text=news)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

