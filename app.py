import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# LOAD MODELS
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_dict = joblib.load("symptom_dict.pkl")

# =========================
# LOAD DATASET (for medicine lookup)
# =========================
df = pd.read_csv("training.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Virtual Doctor",
    page_icon="🩺",
    layout="centered"
)

st.title("AI Powered Disease Predictor")
st.write("I will help you analyze your symptoms and suggest possible disease.")

# =========================
# SESSION STATE
# =========================
if "step" not in st.session_state:
    st.session_state.step = 0

if "data" not in st.session_state:
    st.session_state.data = {
        "age": None,
        "gender": None,
        "categories": [],
        "symptoms": []
    }

# =========================
# SYMPTOM CATEGORIES
# =========================
categories_map = {
    "General": ["itching", "fatigue", "weight_loss", "chills", "fever"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "sore_throat"],
    "Digestive": ["nausea", "vomiting", "abdominal_pain", "acidity"],
    "Skin": ["itching", "skin_rash", "blister"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"],
    "Urinary": ["burning_micturition", "urination_frequent"],
    "Cardiovascular": ["chest_pain", "fast_heart_rate"],
    "Musculoskeletal": ["joint_pain", "muscle_pain"]
}

# =========================
# STEP 1 — AGE
# =========================
if st.session_state.step == 0:
    st.subheader("Question 1: What is your age?")
    age = st.number_input("Enter age", 0, 120)

    if st.button("Next"):
        st.session_state.data["age"] = age
        st.session_state.step += 1

# =========================
# STEP 2 — GENDER
# =========================
elif st.session_state.step == 1:
    st.subheader("Question 2: What is your gender?")
    gender = st.radio("Select", ["Male", "Female"])

    if st.button("Next"):
        st.session_state.data["gender"] = gender
        st.session_state.step += 1

# =========================
# STEP 3 — CATEGORY
# =========================
elif st.session_state.step == 2:
    st.subheader("Select symptom categories")

    selected_categories = st.multiselect(
        "Choose categories",
        list(categories_map.keys())
    )

    if st.button("Next"):
        st.session_state.data["categories"] = selected_categories
        st.session_state.step += 1

# =========================
# STEP 4 — SYMPTOMS
# =========================
elif st.session_state.step == 3:
    st.subheader("Select your symptoms")

    available_symptoms = []

    for cat in st.session_state.data["categories"]:
        available_symptoms.extend(categories_map[cat])

    available_symptoms = list(set(available_symptoms))

    selected_symptoms = st.multiselect(
        "Choose symptoms",
        available_symptoms
    )

    if st.button("Predict Disease"):
        st.session_state.data["symptoms"] = selected_symptoms

        # =========================
        # BUILD FEATURE VECTOR
        # =========================
        features = np.zeros(len(symptom_dict))

        for sym in selected_symptoms:
            if sym in symptom_dict:
                features[symptom_dict[sym]] = 1

        # =========================
        # PREDICTION
        # =========================
        pred = model.predict([features])[0]
        disease = encoder.inverse_transform([pred])[0]

        # =========================
        # GET MEDICINE FROM DATASET
        # =========================
        medicine_row = df[df["prognosis"] == disease]

        if not medicine_row.empty:
            medicine = medicine_row.iloc[0]["medicine"]
        else:
            medicine = "Consult doctor for proper medication."

        # =========================
        # OUTPUT UI
        # =========================
        st.success("🩺 Diagnosis Complete")

        st.subheader("Predicted Disease")
        st.write(disease)

        st.subheader("💊 Recommended Medicine")
        st.write(medicine)

        st.subheader("🛡️ Precautions")
        st.write("""
        - Maintain hygiene  
        - Drink plenty of water  
        - Rest properly  
        - Avoid self-medication  
        - Consult a doctor if symptoms worsen  
        """)