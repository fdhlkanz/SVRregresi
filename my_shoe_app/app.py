import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# === Load Model ===
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# === Load OneHotEncoder & Scaler jika diperlukan ===
with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        brand = request.form.get("brand")
        sold = float(request.form.get("sold"))
        rating = float(request.form.get("rating"))

        # Preprocessing sama seperti training
        brand_encoded = encoder.transform([[brand]]).toarray()
        numeric_scaled = scaler.transform([[sold, rating]])
        X_input = np.concatenate([numeric_scaled, brand_encoded], axis=1)

        # Prediksi
        prediction = model.predict(X_input)[0]
        return render_template(
            "index.html",
            prediction_text=f"Prediksi harga sepatu: ₹{prediction:,.2f}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
