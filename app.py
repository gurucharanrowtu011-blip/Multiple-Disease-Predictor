import streamlit as st
import numpy as np
import joblib
import pandas as pd

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
# SYMPTOM LIST (MUST MATCH TRAINING ORDER)
# =========================
symptoms = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
    'burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
    'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
    'extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain',
    'hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
    'spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine',
    'passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability',
    'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite',
    'polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
    'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
    'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister',
    'red_sore_around_nose','yellow_crust_ooze','Urinating_a_lot','Heartburn'
]

# =========================
# CATEGORY MAP (CLEAN UI)
# =========================
categories = {
    "🔥 General": ["fatigue","lethargy","malaise","weight_loss","weight_gain","loss_of_appetite","dehydration"],
    "🤒 Fever/ Infection": ["high_fever","mild_fever","chills","shivering","sweating"],
    "🫁 Respiratory": ["cough","phlegm","breathlessness","chest_pain","congestion","runny_nose"],
    "🍽️ Digestive": ["stomach_pain","vomiting","nausea","diarrhoea","acidity","indigestion"],
    "🧠 Neurological": ["headache","dizziness","loss_of_balance","slurred_speech","altered_sensorium"],
    "🧴 Skin": ["itching","skin_rash","blister","pus_filled_pimples","blackheads","skin_peeling"],
    "💧 Urinary": ["burning_micturition","continuous_feel_of_urine","dark_urine","polyuria"],
    "❤️ Cardio": ["chest_pain","palpitations","fast_heart_rate"],
    "🦴 Musculoskeletal": ["joint_pain","back_pain","neck_pain","knee_pain","muscle_weakness"],
    "🧬 Systemic": ["obesity","enlarged_thyroid","irregular_sugar_level"]
}

# =========================
# UI INPUT
# =========================
st.subheader("Select Symptoms by Category")

selected_symptoms = []

for cat, syms in categories.items():
    with st.expander(cat):
        for s in syms:
            if s in symptoms:
               if st.checkbox(s, key=cat + "_" + s):
                   selected_symptoms.append(s)

# =========================
# PREDICTION
# =========================
def predict(symptoms_selected):
    input_vector = np.zeros(len(symptoms))

    for s in symptoms_selected:
        if s in symptoms:
            idx = symptoms.index(s)
            input_vector[idx] = 1

    pred = model.predict([input_vector])[0]
    disease = encoder.inverse_transform([pred])[0]

    return disease

# =========================
# BUTTON
# =========================
if st.button("🩺 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:
        disease = predict(selected_symptoms)

        st.success(f"🧾 Disease: {disease}")

        st.info("💊 Medicines & precautions will be added from dataset mapping (next upgrade)")
