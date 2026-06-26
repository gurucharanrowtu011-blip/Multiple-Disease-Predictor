import streamlit as st
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="wide")
st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL + ENCODER
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("disease_predictor_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# =========================
# FULL SYMPTOM LIST (MUST MATCH TRAINING ORDER)
# =========================
symptoms = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
    'burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','breathlessness',
    'sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine',
    'nausea','loss_of_appetite','back_pain','constipation','abdominal_pain','diarrhoea',
    'mild_fever','yellow_urine','yellowing_of_eyes','phlegm','throat_irritation',
    'runny_nose','congestion','chest_pain','fast_heart_rate','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','loss_of_balance','bladder_discomfort','continuous_feel_of_urine',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
    'increased_appetite','polyuria','family_history','mucoid_sputum','lack_of_concentration',
    'visual_disturbances','coma','stomach_bleeding','distention_of_abdomen',
    'blood_in_sputum','palpitations','painful_walking','blackheads','skin_peeling',
    'blister','pus_filled_pimples','yellow_crust_ooze','loss_of_smell'
]

# =========================
# CATEGORY SYSTEM (MORE COMPLETE)
# =========================
categories = {
    "🔥 General": [
        "fatigue","lethargy","malaise","weight_loss","weight_gain","loss_of_appetite",
        "dehydration","weakness_in_limbs","restlessness","anxiety","depression","irritability"
    ],
    "🤒 Fever / Infection": [
        "high_fever","mild_fever","chills","shivering","sweating","toxic_look_(typhos)"
    ],
    "🫁 Respiratory": [
        "cough","phlegm","breathlessness","chest_pain","runny_nose",
        "congestion","mucoid_sputum","rusty_sputum","throat_irritation","loss_of_smell"
    ],
    "🍽️ Digestive": [
        "stomach_pain","abdominal_pain","nausea","vomiting","diarrhoea",
        "indigestion","acidity","constipation","loss_of_appetite","stomach_bleeding"
    ],
    "🧠 Neurological": [
        "headache","dizziness","loss_of_balance","slurred_speech",
        "altered_sensorium","visual_disturbances","unsteadiness","spinning_movements"
    ],
    "🧴 Skin": [
        "itching","skin_rash","blister","blackheads","pus_filled_pimples",
        "red_spots_over_body","skin_peeling","yellow_crust_ooze"
    ],
    "💧 Urinary": [
        "burning_micturition","continuous_feel_of_urine","dark_urine",
        "polyuria","bladder_discomfort"
    ],
    "❤️ Cardio": [
        "chest_pain","palpitations","fast_heart_rate","swollen_blood_vessels"
    ],
    "🦴 Musculoskeletal": [
        "joint_pain","back_pain","neck_pain","knee_pain",
        "muscle_pain","muscle_weakness","stiff_neck","swelling_joints"
    ],
    "🧬 Systemic / Metabolic": [
        "obesity","enlarged_thyroid","irregular_sugar_level",
        "increased_appetite","polyuria","family_history"
    ]
}

# =========================
# UI (NO DUPLICATE IDS FIXED)
# =========================
st.subheader("Select Symptoms")

selected_symptoms = []

for cat, syms in categories.items():
    with st.expander(cat):
        for s in syms:
            if s in symptoms:
                key = f"{cat}_{s}"   # FIX: avoids duplicate checkbox error
                if st.checkbox(s, key=key):
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

    pred_index = model.predict([input_vector])[0]

    # IMPORTANT FIX: convert encoded label → disease name
    disease_name = encoder.inverse_transform([pred_index])[0]

    return disease_name

# =========================
# BUTTON
# =========================
if st.button("🩺 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Select at least one symptom")
    else:
        disease = predict(selected_symptoms)
        st.success(f"🧾 Disease: {disease}")
