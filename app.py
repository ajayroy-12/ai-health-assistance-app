import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os

# --- Configuration ---
st.set_page_config(
    page_title="HealthGuard Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    :root {
        --primary: #4361ee;       /* Vibrant blue */
        --secondary: #3a0ca3;      /* Deep purple-blue */
        --accent: #4cc9f0;         /* Light cyan */
        --danger: #f72585;         /* Pink-red */
        --success: #4ad66d;        /* Fresh green */
        --background: #f8f9fa;     /* Off-white background */
        --text-primary: #2b2d42;   /* Dark blue-gray */
        --text-secondary: #8d99ae; /* Medium gray */
        
        /* Shadow variables */
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .chat-user {
        background-color: var(--primary);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: 25%;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        position: relative;
        line-height: 1.5;
    }
    
    .chat-user:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .chat-user::after {
        content: "";
        position: absolute;
        right: -8px;
        top: 12px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-left-color: var(--primary);
        border-right: 0;
    }
    
    .chat-bot {
        background-color: white;
        color: var(--text-primary);
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 75%;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        position: relative;
        line-height: 1.5;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .chat-bot:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .chat-bot::before {
        content: "";
        position: absolute;
        left: -8px;
        top: 12px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-right-color: white;
        border-left: 0;
    }
    
    /* Message timestamp */
    .message-time {
        display: block;
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-top: 4px;
        text-align: right;
    }
    
    /* For the chat container */
    .chat-container {
        background-color: var(--background);
        padding: 20px;
        border-radius: 12px;
        box-shadow: var(--shadow-lg);
    }
</style>
    """, unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def load_models():
    models = {
        'diabetes': joblib.load('diabetes_model.pkl'),
        'heart': joblib.load('heart_model.pkl'),
        'diabetes_scaler': joblib.load('diabetes_scaler.pkl'),
        'heart_scaler':joblib.load('heart_scaler.pkl')
    }
    return models

models = load_models()

# --- Image Loader ---
def load_images():
    images = {}
    image_files = {
        'home': 'images/home_banner.jpg',
        'diabetes': 'images/diabetes_icon.jpg',
        'heart': 'images/heart_icon.jpeg',
        'bmi': 'images/bmi_icon.png'
    }
    
    for name, path in image_files.items():
        try:
            images[name] = Image.open(path)
        except:
            images[name] = None
    return images

images = load_images()

# --- Chatbot ---
class HealthChatbot:
    def __init__(self):
        self.knowledge_base = {
            "diabetes": {
                "symptoms": ["thirst", "urinate", "hunger", "fatigue", "blurry vision"],
                "advice": "These could be diabetes symptoms. Check your glucose levels and consider our diabetes assessment."
            },
            "heart": {
                "symptoms": ["chest pain", "shortness of breath", "nausea", "fatigue"],
                "advice": "These may indicate heart issues. Try our heart disease assessment and consult a doctor if symptoms persist."
            },
            "kidney": {
                "symptoms": ["swelling", "fatigue", "urination", "back pain"],
                "advice": "Possible kidney health issues."
            },
            "general": {
                "hi": "Hello! I'm your health assistant. How can I help?",
                "help": "I can help assess risks for diabetes, heart disease, and kidney health. Just describe your symptoms.",
                "thanks": "You're welcome! Stay healthy!"
            }
        }
    
    def respond(self, user_input):
        user_input = user_input.lower()
        
        # Check for exact matches
        for category in self.knowledge_base.values():
            if user_input in category:
                return category[user_input]
        
        # Check for symptom keywords
        for condition, data in self.knowledge_base.items():
            if condition != "general":
                for symptom in data["symptoms"]:
                    if symptom in user_input:
                        return data["advice"]
        
        # Default response
        return "I'm not sure I understand. Could you describe your symptoms more specifically?"

# --- Main App ---
def main():
    st.sidebar.title("HealthGuard Pro")
    app_mode = st.sidebar.radio("Navigation", 
        ["🏠 Home", "🩺 Health Assessment", "📊 BMI Calculator", "💬 Health Chat"])
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "🩺 Health Assessment":
        show_assessment()
    elif app_mode == "📊 BMI Calculator":
        show_bmi_calculator()
    elif app_mode == "💬 Health Chat":
        show_chat()

# --- Chat Page ---
def show_chat():
    st.title("Health Assistant Chat")
    
    # Initialize chat history and bot
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'bot' not in st.session_state:
        st.session_state.bot = HealthChatbot()

    if 'reset_input' not in st.session_state:
        st.session_state.reset_input = False

    # Clear input BEFORE rendering widget
    if st.session_state.reset_input:
        st.session_state.chat_input = ""
        st.session_state.reset_input = False

    # Display chat history
    for speaker, text in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f'<div class="chat-user">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">{text}</div>', unsafe_allow_html=True)

    # User input
    user_input = st.text_input("Type your health question...", key="chat_input")

    if user_input and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        
        # Get bot response
        response = st.session_state.bot.respond(user_input)
        st.session_state.chat_history.append(("bot", response))

        # Set flag to clear input and rerun
        st.session_state.reset_input = True
        st.rerun()

# [Rest of your existing code for other pages...]

# --- Home Page ---
def show_home():
    st.title("Welcome to HealthGuard Pro")
    st.markdown("Your comprehensive health risk assessment tool")
    
    if images['home']:
        st.image(images['home'], use_column_width=True)
    
    st.markdown("""
    ## About HealthGuard Pro (🩺AI-POWERED PERSONAL HEALTH ASSISTANCE)
    This application helps you assess your risk for several health conditions:
    - **Diabetes Prediction**: Evaluate your risk of developing diabetes
    - **Heart Disease Assessment**: Check your cardiovascular health
    - **BMI Calculator**: Track your body mass index
                
    Use the navigation menu on the left to access different tools.
    """)
    
    # Features section
    st.header("Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if images['diabetes']:
            st.image(images['diabetes'], width=100)
        st.subheader("Diabetes Assessment")
        st.write("Predict your risk of diabetes based on key health indicators")
    
    with col2:
        if images['heart']:
            st.image(images['heart'], width=100)
        st.subheader("Heart Health")
        st.write("Evaluate your cardiovascular health and risk factors")
    

# --- Health Assessment Page ---
def show_assessment():
    st.title("Health Risk Assessment")
    assessment_type = st.selectbox(
        "Select Assessment Type",
        ["Diabetes", "Heart Disease"]
    )
    
    if assessment_type == "Diabetes":
        st.subheader("Diabetes Risk Assessment")
        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
                glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 100)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
                skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            
            with col2:
                insulin = st.number_input("Insulin Level (μU/mL)", 0, 1000, 80)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
                age = st.number_input("Age", 0, 120, 30)
            
            submitted = st.form_submit_button("Assess Risk")
            
            if submitted:
                input_data = np.array([[
                    pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, diabetes_pedigree, age
                ]])
                
                # Scale the data
                scaled_data = models['diabetes_scaler'].transform(input_data)
                
                # Make prediction
                prediction = models['diabetes'].predict(scaled_data)
                probability = models['diabetes'].predict_proba(scaled_data)[0][1]
                
                if prediction[0] == 1:
                    st.error(f"High risk of diabetes ({probability*100:.1f}% probability)")
                    st.write("Consider consulting a doctor and making lifestyle changes.")
                else:
                    st.success(f"Low risk of diabetes ({probability*100:.1f}% probability)")
                    st.write("Maintain healthy habits to keep your risk low.")
    
    elif assessment_type == "Heart Disease":
        st.subheader("Heart Disease Risk Assessment")
        with st.form("heart_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 0, 120, 50)
                sex = st.selectbox("Sex", ["Male", "Female"])
                cp = st.selectbox("Chest Pain Type", [
                    "Typical angina", 
                    "Atypical angina", 
                    "Non-anginal pain", 
                    "Asymptomatic"
                ])
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
            
            with col2:
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
                restecg = st.selectbox("Resting ECG Results", [
                    "Normal",
                    "ST-T wave abnormality",
                    "Left ventricular hypertrophy"
                ])
                thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
                exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
                slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ])
                ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
                thal = st.selectbox("Thalassemia", [
                    "Normal",
                    "Fixed defect",
                    "Reversible defect"
                ])
            
            submitted = st.form_submit_button("Assess Risk")
            
            if submitted:
                # Convert inputs to model format
                sex = 1 if sex == "Male" else 0
                cp_map = {
                    "Typical angina": 0,
                    "Atypical angina": 1,
                    "Non-anginal pain": 2,
                    "Asymptomatic": 3
                }
                cp = cp_map[cp]
                
                fbs = 1 if fbs == "Yes" else 0
                
                restecg_map = {
                    "Normal": 0,
                    "ST-T wave abnormality": 1,
                    "Left ventricular hypertrophy": 2
                }
                restecg = restecg_map[restecg]
                
                exang = 1 if exang == "Yes" else 0
                
                slope_map = {
                    "Upsloping": 0,
                    "Flat": 1,
                    "Downsloping": 2
                }
                slope = slope_map[slope]
                
                thal_map = {
                    "Normal": 0,
                    "Fixed defect": 1,
                    "Reversible defect": 2
                }
                thal = thal_map[thal]
                
                input_data = np.array([[
                    age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal
                ]])
                
                # Scale the data
                scaled_data = models['heart_scaler'].transform(input_data)
                
                # Make prediction
                prediction = models['heart'].predict(scaled_data)
                probability = models['heart'].predict_proba(scaled_data)[0][1]
                
                if prediction[0] == 1:
                    st.error(f"High risk of heart disease ({probability*100:.1f}% probability)")
                    st.write("Please consult a cardiologist for further evaluation.")
                else:
                    st.success(f"Low risk of heart disease ({probability*100:.1f}% probability)")
                    st.write("Continue maintaining heart-healthy habits.")
    
# --- BMI Calculator Page ---
def show_bmi_calculator():
    st.title("BMI Calculator")
    
    if images['bmi']:
        st.image(images['bmi'], width=150)
    
    with st.form("bmi_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            height_unit = st.radio("Height Unit", ["cm", "feet"])
            if height_unit == "cm":
                height = st.number_input("Height (cm)", 50, 300, 170)
            else:
                feet = st.number_input("Feet", 1, 8, 5)
                inches = st.number_input("Inches", 0, 11, 6)
                height = feet * 30.48 + inches * 2.54
        
        with col2:
            weight_unit = st.radio("Weight Unit", ["kg", "pounds"])
            if weight_unit == "kg":
                weight = st.number_input("Weight (kg)", 10, 300, 70)
            else:
                pounds = st.number_input("Weight (pounds)", 20, 660, 150)
                weight = pounds * 0.453592
        
        submitted = st.form_submit_button("Calculate BMI")
        
        if submitted:
            bmi = weight / ((height/100) ** 2)
            st.subheader(f"Your BMI: {bmi:.1f}")
            
            if bmi < 18.5:
                st.warning("Underweight")
                st.write("Consider consulting a nutritionist for healthy weight gain.")
            elif 18.5 <= bmi < 25:
                st.success("Normal weight")
                st.write("Maintain your healthy lifestyle!")
            elif 25 <= bmi < 30:
                st.warning("Overweight")
                st.write("Consider increasing physical activity and improving diet.")
            else:
                st.error("Obese")
                st.write("Please consult with a healthcare provider for weight management options.")

if __name__ == "__main__":
    main()