import streamlit as st
import pandas as pd
import joblib


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Accident Risk Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================
# Load Model and Features
# =========================

@st.cache_resource
def load_model():

    model = joblib.load(
        "final_tuned_xgboost_model.pkl"
    )

    feature_columns = joblib.load(
        "feature_columns.pkl"
    )

    return model, feature_columns


model, feature_columns = load_model()


# =========================
# Title
# =========================

st.title("🚗 Accident Risk Prediction")

st.write(
    "Enter road and environmental information "
    "to predict the accident risk."
)


# =========================
# Input Fields
# =========================

col1, col2 = st.columns(2)


with col1:

    num_lanes = st.number_input(
        "Number of Lanes",
        min_value=1,
        max_value=10,
        value=2
    )

    curvature = st.number_input(
        "Curvature",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    speed_limit = st.number_input(
        "Speed Limit",
        min_value=10,
        max_value=200,
        value=60
    )

    num_reported_accidents = st.number_input(
        "Number of Reported Accidents",
        min_value=0,
        max_value=100,
        value=5
    )


with col2:

    road_type = st.selectbox(
        "Road Type",
        [
            "urban",
            "rural",
            "highway"
        ]
    )

    lighting = st.selectbox(
        "Lighting",
        [
            "daylight",
            "dim",
            "night"
        ]
    )

    weather = st.selectbox(
        "Weather",
        [
            "rainy",
            "clear",
            "foggy"
        ]
    )

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "afternoon",
            "evening",
            "morning"
        ]
    )


# =========================
# Boolean Features
# =========================

road_signs_present = st.checkbox(
    "Road Signs Present"
)

public_road = st.checkbox(
    "Public Road"
)

holiday = st.checkbox(
    "Holiday"
)

school_season = st.checkbox(
    "School Season"
)


# =========================
# Prediction
# =========================

if st.button("Predict Accident Risk"):

    # =========================
    # Create Input DataFrame
    # =========================

    input_data = pd.DataFrame({

        "num_lanes": [num_lanes],

        "curvature": [curvature],

        "speed_limit": [speed_limit],

        "num_reported_accidents": [
            num_reported_accidents
        ],

        "road_signs_present": [
            int(road_signs_present)
        ],

        "public_road": [
            int(public_road)
        ],

        "holiday": [
            int(holiday)
        ],

        "school_season": [
            int(school_season)
        ],

        "road_type": [
            road_type
        ],

        "lighting": [
            lighting
        ],

        "weather": [
            weather
        ],

        "time_of_day": [
            time_of_day
        ]

    })


    # =========================
    # Preprocessing
    # =========================

    categorical_cols = [

        "road_type",

        "lighting",

        "weather",

        "time_of_day"

    ]


    input_encoded = pd.get_dummies(

        input_data,

        columns=categorical_cols,

        drop_first=True

    )


    # =========================
    # Align Features
    # =========================

    input_encoded = input_encoded.reindex(

        columns=feature_columns,

        fill_value=0

    )


    # =========================
    # Prediction
    # =========================

    prediction = model.predict(

        input_encoded

    )[0]


    # =========================
    # Display Result
    # =========================

    st.success(
        f"Predicted Accident Risk: {prediction:.4f}"
    )

