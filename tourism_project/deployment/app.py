
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Wellness Tourism Package Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

# --------------------------------------------------
# Load the committed trained model
# --------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "tourism_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# --------------------------------------------------
# Collect customer inputs
# --------------------------------------------------
st.subheader("Customer Details")

with st.form("tourism_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        typeofcontact = st.selectbox(
            "Type of Contact",
            ["Company Invited", "Self Enquiry"]
        )
        citytier = st.selectbox("City Tier", [1, 2, 3])
        durationofpitch = st.number_input(
            "Duration of Pitch (minutes)", 0, 1000, 15
        )
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        numberofpersonvisiting = st.number_input(
            "Number of Persons Visiting", 1, 20, 2
        )
        numberoffollowups = st.number_input(
            "Number of Follow-ups", 1, 20, 3
        )
        productpitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        preferredpropertystar = st.selectbox(
            "Preferred Property Star", [3, 4, 5]
        )
        maritalstatus = st.selectbox(
            "Marital Status",
            ["Married", "Divorced", "Single", "Unmarried"]
        )
        numberoftrips = st.number_input(
            "Number of Trips", 0, 100, 3
        )

    with col3:
        passport = st.selectbox("Has Passport?", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        pitchsatisfactionscore = st.selectbox(
            "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
        )
        owncar = st.selectbox(
            "Owns a Car?", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        numberofchildrenvisiting = st.number_input(
            "Number of Children Visiting", 0, 20, 0
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthlyincome = st.number_input(
            "Monthly Income", min_value=0, value=25000, step=1000
        )

    submitted = st.form_submit_button("Predict Purchase")

# --------------------------------------------------
# Create a DataFrame and predict
# --------------------------------------------------
if submitted:

    input_data = {
        "Age": [age],
        "TypeofContact": [typeofcontact],
        "CityTier": [citytier],
        "DurationOfPitch": [durationofpitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [numberofpersonvisiting],
        "NumberOfFollowups": [numberoffollowups],
        "ProductPitched": [productpitched],
        "PreferredPropertyStar": [preferredpropertystar],
        "MaritalStatus": [maritalstatus],
        "NumberOfTrips": [numberoftrips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitchsatisfactionscore],
        "OwnCar": [owncar],
        "NumberOfChildrenVisiting": [numberofchildrenvisiting],
        "Designation": [designation],
        "MonthlyIncome": [monthlyincome],
    }

    input_df = pd.DataFrame(input_data)

    st.subheader("Customer Input Data")
    st.dataframe(input_df, use_container_width=True)

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.success(
                "The model predicts that the customer is likely "
                "to purchase the Wellness Tourism Package."
            )
        else:
            st.info(
                "The model predicts that the customer is unlikely "
                "to purchase the Wellness Tourism Package."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.metric(
                "Estimated purchase probability",
                f"{probabilities[1] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "Check that the deployed model was trained with the same "
            "feature names and preprocessing as this app."
        )

