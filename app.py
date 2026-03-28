
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# Download and load the model from your Hugging Face repo
model_path = hf_hub_download(
    repo_id="Murali0606/tourism-model",   # your repo name
    filename="tourism_best_model.pkl"          # the file you saved in the repo
)
model = joblib.load(model_path)

st.title("Tourism Prediction Demo")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
family_members = st.number_input("Family Members", min_value=1, max_value=10, value=4)
duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=7)

if st.button("Predict"):
    input_data = [[age, income, family_members, duration]]
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted tourism category: {prediction}")




