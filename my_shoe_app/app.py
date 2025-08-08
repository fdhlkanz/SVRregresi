import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import os

# === Load Model dan Preprocessing ===
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    ohe = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# === Init Flask App ===
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        brand = request.form.get("brand")
        sold = float(request.form.get("sold"))
        rating = float(request.form.get("rating"))

        # --- Preprocessing ---
        brand_encoded = ohe.transform([[brand]])
        numeric_scaled = scaler.transform([[sold, rating]])
        X_input = np.concatenate([numeric_scaled, brand_encoded], axis=1)

        # --- Predict ---
        prediction = model.predict(X_input)[0]

        return render_template("index.html",
                               prediction_text=f"Prediksi harga sepatu: ₹{prediction:,.2f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
