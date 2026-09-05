import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Must match the threshold used in train.py
CLASSIFICATION_THRESHOLD = 0.45

st.set_page_config(page_title="Wellness Package Prediction", page_icon="🧳")

st.title("Wellness Tourism Package — Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the newly introduced
**Wellness Tourism Package**. Enter the customer profile and sales pitch details below to get a
prediction, so the team can prioritise outreach before making contact.
""")

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status",
                                  ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation",
                              ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation",
                               ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 22000, step=500)

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3],
                             help="Tier 1 is the most developed")
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    num_persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    num_children = st.number_input("Number of Children Visiting (under 5)", 0, 5, 1)
    num_trips = st.number_input("Number of Trips per Year", 0, 25, 3)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

st.header("Interaction Details")
col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox("Product Pitched",
                                   ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)
    num_followups = st.number_input("Number of Followups", 0, 10, 4)

with col4:
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    passport = st.selectbox("Holds a Valid Passport", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car", ["Yes", "No"])

# Assemble a single-row DataFrame with the exact column names used in training.
# Values are passed raw - the saved pipeline handles imputation, scaling and encoding.
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Show the row being sent to the model"):
    st.dataframe(input_data)

if st.button("Predict Purchase", type="primary"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"**Likely to purchase** the Wellness Tourism Package "
                   f"(probability {probability:.1%})")
        st.write("Recommended action: prioritise this customer for outreach.")
    else:
        st.warning(f"**Unlikely to purchase** the Wellness Tourism Package "
                   f"(probability {probability:.1%})")
        st.write("Recommended action: deprioritise, or target with a different package.")

    st.progress(float(probability))
    st.caption(f"Classification threshold: {CLASSIFICATION_THRESHOLD}. "
               "Probabilities above this are flagged as likely purchasers.")
