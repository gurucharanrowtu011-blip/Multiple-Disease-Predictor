import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")

st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    return joblib.load("disease_predictor_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
encoder = load_encoder()

# =========================
# LOAD DATA (medicine lookup)
# =========================
df = pd.read_csv("training.csv")

# clean column names
df.columns = df.columns.str.strip()

# =========================
# SYMPTOM CATEGORIES
# =========================
categories = {
    # 1. GENERAL / CONSTITUTIONAL
    "General": [
        "fatigue", "lethargy", "malaise", "restlessness",
        "anxiety", "depression", "irritability",
        "loss_of_appetite", "weight_gain", "weight_loss",
        "dehydration"
    ],

    # 2. FEVER / INFECTION
    "Fever / Infection": [
        "high_fever", "mild_fever", "chills", "shivering",
        "sweating", "toxic_look_(typhos)"
    ],

    # 3. RESPIRATORY
    "Respiratory": [
        "cough", "phlegm", "mucoid_sputum", "rusty_sputum",
        "throat_irritation", "runny_nose", "congestion",
        "breathlessness", "chest_pain", "fast_heart_rate",
        "blood_in_sputum", "loss_of_smell"
    ],

    # 4. DIGESTIVE
    "Digestive": [
        "stomach_pain", "abdominal_pain", "constipation",
        "diarrhoea", "indigestion", "nausea", "vomiting",
        "acidity", "ulcers_on_tongue", "heartburn",
        "bloody_stool", "passage_of_gases"
    ],

    # 5. NEUROLOGICAL
    "Neurological": [
        "headache", "dizziness", "spinning_movements",
        "loss_of_balance", "unsteadiness", "slurred_speech",
        "altered_sensorium", "weakness_of_one_body_side",
        "visual_disturbances", "blurred_and_distorted_vision"
    ],

    # 6. SKIN / DERMATOLOGICAL
    "Skin": [
        "itching", "skin_rash", "nodal_skin_eruptions",
        "red_spots_over_body", "pus_filled_pimples",
        "blackheads", "scurring", "skin_peeling",
        "blister", "red_sore_around_nose",
        "yellow_crust_ooze", "bruising"
    ],

    # 7. URINARY / KIDNEY
    "Urinary": [
        "burning_micturition", "continuous_feel_of_urine",
        "bladder_discomfort", "polyuria",
        "Urinating_a_lot", "foul_smell_of urine",
        "dark_urine", "yellow_urine", "spotting_ urination"
    ],

    # 8. CARDIOVASCULAR
    "Cardiovascular": [
        "chest_pain", "palpitations", "fast_heart_rate",
        "swollen_blood_vessels", "prominent_veins_on_calf"
    ],

    # 9. MUSCULOSKELETAL
    "Musculoskeletal": [
        "joint_pain", "back_pain", "neck_pain", "knee_pain",
        "hip_joint_pain", "muscle_pain", "muscle_weakness",
        "swelling_joints", "movement_stiffness",
        "painful_walking", "cramps"
    ],

    # 10. METABOLIC / ENDOCRINE
    "Metabolic": [
        "obesity", "enlarged_thyroid",
        "irregular_sugar_level",
        "increased_appetite",
        "weight_gain", "weight_loss"
    ]
}

# =========================
# SESSION STATE
# =========================
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.symptoms = []
    st.session_state.age = None
    st.session_state.gender = None

# =========================
# RESET
# =========================
def reset():
    st.session_state.step = 0
    st.session_state.symptoms = []
    st.session_state.age = None
    st.session_state.gender = None

# =========================
# BUILD FEATURE VECTOR
# =========================
def build_vector(selected):
    features = df.columns[:-2]  # last 2 = prognosis, medicine
    vec = np.zeros(len(features))

    for s in selected:
        if s in features:
            idx = list(features).index(s)
            vec[idx] = 1

    return np.array(vec).reshape(1, -1)

# =========================
# PREDICT FUNCTION
# =========================
def predict(symptoms):
    X = build_vector(symptoms)

    pred_index = model.predict(X)[0]
    disease = encoder.inverse_transform([pred_index])[0]

    medicine = df[df["prognosis"] == disease]["medicine"].values
    medicine = medicine[0] if len(medicine) > 0 else "Not Available"

    return disease, medicine

# =========================
# UI FLOW
# =========================

# STEP 0 - AGE
if st.session_state.step == 0:
    st.subheader("What is your age?")
    age = st.number_input("Age", 1, 120)

    if st.button("Next"):
        st.session_state.age = age
        st.session_state.step = 1

# STEP 1 - GENDER
elif st.session_state.step == 1:
    st.subheader("Select Gender")
    gender = st.radio("Gender", ["Male", "Female"])

    if st.button("Next"):
        st.session_state.gender = gender
        st.session_state.step = 2

# STEP 2 - CATEGORY
elif st.session_state.step == 2:
    st.subheader("Select Symptom Category")

    category = st.selectbox("Category", list(categories.keys()))

    symptoms = st.multiselect("Select Symptoms", categories[category])

    if st.button("Next"):
        st.session_state.symptoms.extend(symptoms)
        st.session_state.step = 3

# STEP 3 - RESULT
elif st.session_state.step == 3:
    st.subheader("🩺 Diagnosis")

    if len(st.session_state.symptoms) == 0:
        st.warning("No symptoms selected")
        st.stop()

    disease, medicine = predict(st.session_state.symptoms)

    st.success(f"Predicted Disease: {disease}")
    st.info(f"Recommended Medicine: {medicine}")

    if st.button("Restart"):
        reset()
