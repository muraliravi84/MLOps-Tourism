
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Murali0606/tourism-model",   # replace with your repo name
    filename="tourism_best_model.pkl"     # replace with your model filename
)
model = joblib.load(model_path)

st.title("Tourism Prediction Demo")

# Collect all 18 features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
family_members = st.number_input("Family Members", min_value=1, max_value=10, value=4)
duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=7)

destination = st.selectbox("Destination Type", ["Beach", "Mountain", "City", "Cultural"])
season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])
transport = st.selectbox("Transport Mode", ["Car", "Train", "Flight", "Bus"])
accommodation = st.selectbox("Accommodation Type", ["Hotel", "Resort", "Homestay", "Hostel"])
purpose = st.selectbox("Purpose of Travel", ["Leisure", "Business", "Pilgrimage", "Education"])
food_pref = st.selectbox("Food Preference", ["Vegetarian", "Non-Vegetarian", "Mixed"])
shopping = st.selectbox("Shopping Interest", ["High", "Medium", "Low"])
adventure = st.selectbox("Adventure Interest", ["Yes", "No"])
cultural_interest = st.selectbox("Cultural Interest", ["Yes", "No"])
kids = st.selectbox("Travelling with Kids", ["Yes", "No"])
elderly = st.selectbox("Travelling with Elderly", ["Yes", "No"])
budget = st.number_input("Budget (INR)", min_value=1000, value=20000)
rating = st.slider("Past Trip Satisfaction (1-5)", 1, 5, 3)
repeat_travel = st.selectbox("Repeat Traveller", ["Yes", "No"])

# Build input data in the same order as training
input_data = [[
    age, income, family_members, duration,
    destination, season, transport, accommodation,
    purpose, food_pref, shopping, adventure,
    cultural_interest, kids, elderly, budget,
    rating, repeat_travel
]]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted tourism category: {prediction}")




