import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Load model
model_path = hf_hub_download(
    repo_id="Murali0606/tourism-model",
    filename="tourism_best_model.pkl"
)
model = joblib.load(model_path)

st.title("Tourism Prediction Demo")

# Collect inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, value=10)
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, value=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, value=1)
passport = st.selectbox("Passport", [0, 1])
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Other"])
monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)

# Manual encoding (must match training LabelEncoder mappings!)
typeof_contact_map = {"Company Invited": 0, "Self Inquiry": 1}
occupation_map = {"Salaried": 0, "Freelancer": 1, "Other": 2}
gender_map = {"Male": 0, "Female": 1}
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
designation_map = {"Manager": 0, "Executive": 1, "Other": 2}
product_map = {"Basic": 0, "Standard": 1, "Deluxe": 2}

typeof_contact_encoded = typeof_contact_map[typeof_contact]
occupation_encoded = occupation_map[occupation]
gender_encoded = gender_map[gender]
marital_status_encoded = marital_status_map[marital_status]
designation_encoded = designation_map[designation]
product_encoded = product_map[product_pitched]

# Build DataFrame with exact feature names and order
input_df = pd.DataFrame([[
    age, typeof_contact_encoded, city_tier, duration_pitch, occupation_encoded,
    gender_encoded, num_person_visiting, num_followups, product_encoded,
    preferred_property_star, marital_status_encoded, num_trips, passport,
    pitch_score, own_car, num_children_visiting, designation_encoded,
    monthly_income
]], columns=[
    'Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender',
    'NumberOfPersonVisiting','NumberOfFollowups','ProductPitched','PreferredPropertyStar',
    'MaritalStatus','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','Designation','MonthlyIncome'
])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted: {'Package Purchased' if prediction==1 else 'No Purchase'} "
               f"(probability of purchase: {probability:.2f})")

