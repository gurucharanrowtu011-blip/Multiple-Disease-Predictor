import streamlit as st
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Multiple Disease Predictor", layout="wide")
st.title("🩺 Multiple Disease Predictor")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("disease_predictor_model.pkl")
    return model

model = load_model()

# =========================
# SYMPTOM LIST
# =========================
symptoms = list(model.feature_names_in_)

# =========================
# CATEGORY UI
# =========================
categories = {
    "🔥 General": [
        "fatigue","lethargy","malaise","weakness_in_limbs","weight_gain",
        "weight_loss","loss_of_appetite","dehydration","anxiety",
        "restlessness","mood_swings","irritability","depression",
        "lack_of_concentration","excessive_hunger","family_history",
        "sweating","high_fever","mild_fever","chills","shivering"
    ],

    "🤒 Fever / Infection": [
        "high_fever","mild_fever","chills","shivering","sweating",
        "swelled_lymph_nodes","red_spots_over_body","patches_in_throat",
        "continuous_sneezing","runny_nose","congestion",
        "throat_irritation","sinus_pressure","swelling_of_stomach",
        "acute_liver_failure","coma","toxic_look_(typhos)"
    ],

    "🫁 Respiratory": [
        "cough","phlegm","breathlessness","chest_pain","runny_nose",
        "congestion","mucoid_sputum","rusty_sputum","blood_in_sputum",
        "throat_irritation","continuous_sneezing","pain_behind_the_eyes"
    ],

    "🍽️ Digestive": [
        "stomach_pain","abdominal_pain","belly_pain","vomiting","nausea",
        "diarrhoea","constipation","acidity","indigestion",
        "loss_of_appetite","ulcers_on_tongue","passage_of_gases",
        "stomach_bleeding","distention_of_abdomen","yellow_urine",
        "yellowish_skin","yellowing_of_eyes","fluid_overload"
    ],

    "🧠 Neurological": [
        "headache","dizziness","loss_of_balance","unsteadiness",
        "slurred_speech","altered_sensorium","visual_disturbances",
        "blurred_and_distorted_vision","spinning_movements",
        "weakness_of_one_body_side","loss_of_smell","coma",
        "pain_behind_the_eyes"
    ],

    "🧴 Skin": [
        "itching","skin_rash","nodal_skin_eruptions","blister",
        "skin_peeling","pus_filled_pimples","blackheads",
        "scurring","silver_like_dusting","small_dents_in_nails",
        "inflammatory_nails","red_sore_around_nose",
        "yellow_crust_ooze","dischromic _patches"
    ],

    "💧 Urinary": [
        "burning_micturition","dark_urine","yellow_urine",
        "continuous_feel_of_urine","bladder_discomfort",
        "spotting_ urination","polyuria","Urinating_a_lot",
        "blood_in_sputum","foul_smell_of urine"
    ],

    "❤️ Cardio": [
        "chest_pain","fast_heart_rate","palpitations",
        "breathlessness","swollen_legs","swollen_blood_vessels",
        "prominent_veins_on_calf","painful_walking"
    ],

    "🦴 Musculoskeletal": [
        "joint_pain","back_pain","neck_pain","muscle_pain",
        "muscle_weakness","muscle_wasting","knee_pain",
        "hip_joint_pain","stiff_neck","swelling_joints",
        "movement_stiffness","cramps","bruising",
        "swollen_extremeties","puffy_face_and_eyes"
    ],

    "🧬 Metabolic / Hormonal": [
        "obesity","irregular_sugar_level","weight_gain",
        "weight_loss","cold_hands_and_feets",
        "enlarged_thyroid","brittle_nails",
        "yellowish_skin","yellowing_of_eyes",
        "excessive_hunger","history_of_alcohol_consumption",
        "receiving_blood_transfusion",
        "receiving_unsterile_injections"
    ]
}

# =========================
# INPUT UI
# =========================
st.subheader("Select Symptoms")

selected_symptoms = []

for cat, syms in categories.items():
    with st.expander(cat):
        for i, s in enumerate(syms):
            if s in symptoms:
                if st.checkbox(s, key=f"{cat}_{s}_{i}"):
                    selected_symptoms.append(s)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(symptoms_selected):

    input_vector = np.zeros(len(symptoms))

    for s in symptoms_selected:
        if s in symptoms:
            idx = symptoms.index(s)
            input_vector[idx] = 1

    prediction = model.predict([input_vector])[0]

    confidence = np.max(model.predict_proba([input_vector])) * 100

    return prediction, confidence

# =========================
# PREDICT BUTTON
# =========================
if st.button("🩺 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")

    else:

        disease, confidence = predict(selected_symptoms)

        st.success(f"🧾 Disease: {disease}")

        st.info(f"📊 Prediction Confidence: {confidence:.2f}%")
