import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="AnkitkumarMalde/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourist Package Churn Prediction App")
st.write("The Tourist Package Churn Prediction App is an internal staff tool designed to forecast whether customers are likely to churn based on their profile details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Collect user input
Age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("The method by which the customer was contacted", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City category", ["Tier 1","Tier 2", "Tier 3"])
Occupation = st.selectbox("Customer's occupation", ["Free Lancer", "Large Business","Salaried", "Small Business"])
Gender = st.selectbox("Customer's occupation", ["Male","Female"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=1, max_value=10, value=3)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=1, max_value=5, value=4)
MaritalStatus = st.selectbox("Marital status of the customer", ["Divorced", "Married","Single", "Unmarried"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=1, max_value=50, value=3)
Passport = st.selectbox("Customer holds a valid passport", ["Yes", "No"])
OwnCar = st.selectbox("Customer owns a car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0, max_value=10, value=2)
Designation = st.selectbox("Customer's designation in their current organization", ["AVP", "VP","Executive", "Manager", "Senior Manager"])
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=1, max_value=100000, value=30000)

#Customer Interaction Data
PitchSatisfactionScore =  st.number_input("Score indicating the customer's satisfaction with the sales pitch", min_value=1, max_value=5, value=4)
ProductPitched = st.selectbox("Type of product pitched to the customer", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch", min_value=0, max_value=20, value=4)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer", min_value=0, max_value=150, value=20)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': 1 if CityTier == "Tier 1" else 2 if CityTier == "Tier 2" else 3,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus':MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar' : 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting' : NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome' : MonthlyIncome,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'ProductPitched' : ProductPitched,
    'NumberOfFollowups' : NumberOfFollowups,
    'DurationOfPitch' : DurationOfPitch    
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
