
import joblib
import subprocess
import threading
import time
import requests
import streamlit as st
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

# ------------------ Flask Backend ------------------
app = Flask(__name__)

model_path = hf_hub_download(repo_id="Murali0606/tourism-model", filename="tourism_best_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Tourism model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    X = [list(input_data.values())]
    prediction = model.predict(X)[0]
    return jsonify({"ProdTaken": int(prediction)})

def run_flask():
    app.run(host="0.0.0.0", port=7860)

# ------------------ Streamlit Frontend ------------------
def run_streamlit():
    API_URL = "http://localhost:7860/predict"

    st.title("Tourism Model Prediction (via Flask API)")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=50000)
    family_members = st.number_input("Family Members", min_value=1, max_value=10, value=4)
    duration = st.number_input("Duration of Trip (days)", min_value=1, max_value=30, value=7)

    if st.button("Predict"):
        payload = {
            "Age": age,
            "Income": income,
            "FamilyMembers": family_members,
            "Duration": duration
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"ProdTaken: {result['ProdTaken']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ------------------ Run Both ------------------
if __name__ == "__main__":
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Give Flask a moment to start
    time.sleep(2)

    # Run Streamlit (main process)
    run_streamlit()



