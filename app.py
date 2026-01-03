import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import sys

# ============================================
# FIX FOR RENDER: SIMPLIFIED SESSION INITIALIZATION
# ============================================
# Initialize session state at the very beginning
# Use try-except to handle any initialization issues
try:
    # Initialize session state variables
    if 'symptom_input' not in st.session_state:
        st.session_state.symptom_input = ""
    if 'history' not in st.session_state:
        st.session_state.history = []
except:
    # If session state fails, create minimal dict
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    st.session_state.symptom_input = ""
    st.session_state.history = []

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models', exist_ok=True)

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST STREAMLIT COMMAND
# ============================================
st.set_page_config(
    page_title="AI Driven Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    h1, h2, h3 {
        color: #006400;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005a8c;
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #0077b6;
    }
    .metric-card {
        background: linear-gradient(135deg, #e6f7ff 0%, #f0fff0 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-top: 4px solid #0077b6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DISEASE PREDICTOR CLASS
# ============================================
class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        # Basic stopwords list
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        ])
    
    def load_model(self):
        """Load trained model"""
        try:
            model_path = 'models/trained_model.pkl'
            if not os.path.exists(model_path):
                model_path = './models/trained_model.pkl'
                if not os.path.exists(model_path):
                    return False
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            return True
            
        except Exception as e:
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return ' '.join(words)
    
    def predict(self, text):
        """Predict disease from symptoms"""
        if not self.model:
            return None, None, None
        
        try:
            processed = self.preprocess_text(text)
            vectorized = self.vectorizer.transform([processed])
            
            prediction = self.model.predict(vectorized)
            probabilities = self.model.predict_proba(vectorized)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_diseases = self.label_encoder.inverse_transform(top_indices)
            top_probs = probabilities[top_indices]
            
            return self.label_encoder.inverse_transform(prediction)[0], top_diseases, top_probs
            
        except Exception as e:
            return None, None, None

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063812.png", width=100)
st.sidebar.title("🏥 AI Disease Diagnosis")
st.sidebar.markdown("---")

# Navigation - ONLY 4 PAGES
page_options = ["Home", "AI Diagnosis", "Dataset Info", "Train Model"]
page = st.sidebar.radio("Navigate to:", page_options, key="main_navigation")

st.sidebar.markdown("---")
st.sidebar.info("""
**Research Project**
AI-Driven Disease Diagnosis
for Rural Healthcare
""")

# ============================================
# PAGE 1: HOME
# ============================================
if page == "Home":
    st.title("AI-Driven Disease Diagnosis System for Rural Health Care")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        ## Bridging Healthcare Gaps with AI
        
        This system uses machine learning to analyze symptom descriptions
        and suggest possible diseases, designed specifically for 
        resource-constrained healthcare settings.
        
        **Key Features:**
        - Natural language symptom analysis
        - Instant disease predictions
        - Top 3 possible diagnoses with confidence scores
        - Treatment recommendations
        - User-friendly interface
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=200)
    
    # Check model status
    st.markdown("---")
    st.subheader("System Status")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        # Check if model exists
        model_exists = os.path.exists('models/trained_model.pkl') or os.path.exists('./models/trained_model.pkl')
        status_text = "Ready" if model_exists else "Not Ready"
        status_color = "#d4edda" if model_exists else "#fff3cd"
        border_color = "#28a745" if model_exists else "#ffc107"
        
        st.markdown(f"""
        <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color}; text-align: center;">
        <h3>{status_text}</h3>
        <p>AI Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        # Check if dataset exists
        dataset_exists = os.path.exists('Symptom2disease.csv') or os.path.exists('./Symptom2disease.csv')
        status_text = "Ready" if dataset_exists else "Missing"
        status_color = "#d4edda" if dataset_exists else "#f8d7da"
        border_color = "#28a745" if dataset_exists else "#dc3545"
        
        st.markdown(f"""
        <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color}; text-align: center;">
        <h3>{status_text}</h3>
        <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status3:
        st.markdown("""
        <div class="metric-card">
        <h3>4</h3>
        <p>Pages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status4:
        st.markdown("""
        <div class="metric-card">
        <h3>Random Forest</h3>
        <p>Algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("---")
    st.subheader("Quick Start Guide")
    
    with st.expander("How to use this system"):
        st.markdown("""
        **Step 1: Prepare Data**
        - Ensure `Symptom2disease.csv` is in your project folder
        
        **Step 2: Train Model**
        - Go to "Train Model" page
        - Click "Start Training"
        - Wait for completion
        
        **Step 3: Test Diagnosis**
        - Go to "AI Diagnosis" page
        - Enter symptoms
        - Get instant diagnosis
        
        **Requirements:**
        - Python 3.8+
        - Dataset: Symptom2disease.csv
        """)

# ============================================
# PAGE 2: AI DIAGNOSIS
# ============================================
elif page == "AI Diagnosis":
    st.title("AI-Powered Disease Diagnosis")
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Check if model exists
    model_exists = os.path.exists('models/trained_model.pkl') or os.path.exists('./models/trained_model.pkl')
    
    if not model_exists:
        st.error("""
        ## ❌ AI Model Not Trained Yet!
        
        **Please follow these steps:**
        
        1. **Go to "Train Model" page**
        2. **Click "Start Training" button**
        3. **Wait for training to complete**
        4. **Come back to this page**
        
        The model will be created in the `models/` folder.
        """)
        
        if st.button("🔄 Check Again", key="check_again_btn"):
            st.rerun()
        
        st.stop()
    
    # Load model
    with st.spinner("Loading AI model..."):
        if not predictor.load_model():
            st.error("Failed to load model. Please train again.")
            st.stop()
    
    st.success("✅ AI Model loaded successfully!")
    
    # Get disease list
    diseases = predictor.label_encoder.classes_
    
    # ========== DIAGNOSIS INTERFACE ==========
    st.markdown("### 📝 Enter Patient Symptoms")
    
    # Initialize symptom input in session state
    if 'symptom_input' not in st.session_state:
        st.session_state.symptom_input = ""
    
    # Function to update symptoms
    def update_symptoms(symptom):
        if st.session_state.symptom_input:
            st.session_state.symptom_input = f"{st.session_state.symptom_input}, {symptom}"
        else:
            st.session_state.symptom_input = f"I have {symptom}"
    
    # Quick symptom buttons
    st.markdown("**Quick add common symptoms:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Create buttons with on_click callbacks
    if col1.button("Fever", key="btn_fever"):
        update_symptoms("fever")
        st.rerun()
    
    if col2.button("Cough", key="btn_cough"):
        update_symptoms("cough")
        st.rerun()
    
    if col3.button("Headache", key="btn_headache"):
        update_symptoms("headache")
        st.rerun()
    
    if col4.button("Fatigue", key="btn_fatigue"):
        update_symptoms("fatigue")
        st.rerun()
    
    if col5.button("Pain", key="btn_pain"):
        update_symptoms("pain")
        st.rerun()
    
    # Symptom input area
    symptom_text = st.text_area(
        "Describe symptoms in detail:",
        height=150,
        placeholder="Example: I have fever, cough, headache, and fatigue for 2 days...",
        value=st.session_state.symptom_input,
        key="symptom_input_area"
    )
    
    # Update session state
    st.session_state.symptom_input = symptom_text
    
    # Clear button
    if st.button("Clear Symptoms", key="btn_clear"):
        st.session_state.symptom_input = ""
        st.rerun()
    
    # Patient information
    st.markdown("---")
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info2:
        st.markdown("### 👤 Patient Information")
        
        age = st.number_input("Age", 0, 120, 30, key="age_input")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_select")
        duration = st.selectbox("Symptom Duration", 
                               ["<24 hours", "1-3 days", "3-7 days", "1-2 weeks", ">2 weeks"],
                               key="duration_select")
        
        st.markdown("### 🩺 Vital Signs")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0, 0.1, key="temp_input")
            bp_sys = st.number_input("BP Systolic", 80, 200, 120, key="bp_sys_input")
        
        with col_v2:
            hr = st.number_input("Heart Rate", 40, 200, 72, key="hr_input")
            bp_dia = st.number_input("BP Diastolic", 50, 130, 80, key="bp_dia_input")
    
    # Emergency warning
    emergency = False
    if temp > 38.5:
        st.warning("⚠️ High temperature detected (>38.5°C)")
        emergency = True
    if hr > 120:
        st.warning("⚠️ Elevated heart rate (>120 bpm)")
        emergency = True
    if bp_sys > 180 or bp_dia > 110:
        st.warning("⚠️ High blood pressure detected")
        emergency = True
    
    if emergency:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 10px 0;">
        <strong>⚠️ CRITICAL SIGNS DETECTED</strong><br>
        Consider immediate medical consultation.
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnosis button
    st.markdown("---")
    if st.button("🔍 RUN AI DIAGNOSIS", type="primary", use_container_width=True, key="diagnose_btn"):
        current_symptoms = st.session_state.symptom_input
        
        if not current_symptoms.strip():
            st.warning("⚠️ Please enter symptoms to analyze")
        else:
            with st.spinner("🔬 Analyzing symptoms with AI..."):
                # Get prediction
                disease, top_diseases, probs = predictor.predict(current_symptoms)
                
                if disease:
                    # Display results
                    st.markdown("## 📋 Diagnosis Results")
                    
                    confidence = max(probs) * 100
                    
                    # Results columns
                    col_res1, col_res2 = st.columns([2, 1])
                    
                    with col_res1:
                        # Result box
                        if confidence > 80:
                            box_color = "#d4edda"
                            border_color = "#28a745"
                        elif confidence > 60:
                            box_color = "#fff3cd"
                            border_color = "#ffc107"
                        else:
                            box_color = "#f8d7da"
                            border_color = "#dc3545"
                        
                        st.markdown(f"""
                        <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color};">
                        <h3 style="color: #006400;">Primary Diagnosis: {disease}</h3>
                        <p><strong>Confidence Level:</strong> {confidence:.1f}%</p>
                        <p><strong>Patient:</strong> {age}y/o {gender}, symptoms for {duration}</p>
                        <p><strong>Vital Signs:</strong> Temp: {temp}°C, HR: {hr} bpm, BP: {bp_sys}/{bp_dia} mmHg</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top predictions
                        st.markdown("#### Top 3 Predictions:")
                        for d, p in zip(top_diseases, probs):
                            percentage = p * 100
                            st.progress(float(p), text=f"{d}: {percentage:.1f}%")
                    
                    with col_res2:
                        # Create probability chart
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[p*100 for p in probs],
                                y=top_diseases,
                                orientation='h',
                                marker_color=['#28a745' if d == disease else '#0077b6' for d in top_diseases],
                                text=[f'{p*100:.1f}%' for p in probs],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Diagnosis Confidence",
                            xaxis_title="Probability (%)",
                            height=250,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Treatment recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommended Actions")
                    
                    # Treatment mapping
                    treatment_map = {
                        'Psoriasis': [
                            "Use moisturizing creams regularly",
                            "Avoid scratching affected areas",
                            "Consult a dermatologist for proper treatment",
                            "Consider phototherapy if condition is severe"
                        ],
                        'Migraine': [
                            "Rest in a dark, quiet room",
                            "Stay well hydrated",
                            "Avoid known triggers (bright lights, loud noises)",
                            "Consider over-the-counter pain relief if appropriate"
                        ],
                        'Acne': [
                            "Gentle cleansing twice daily",
                            "Avoid picking or squeezing lesions",
                            "Use non-comedogenic skincare products",
                            "Consult dermatologist for persistent cases"
                        ],
                        'Varicose veins': [
                            "Elevate legs when resting",
                            "Wear compression stockings",
                            "Regular walking exercise",
                            "Avoid prolonged standing or sitting"
                        ],
                        'Typhoid': [
                            "Antibiotic treatment as prescribed",
                            "Maintain proper hydration",
                            "Rest and monitor fever pattern",
                            "Follow-up blood tests as recommended"
                        ],
                        'Malaria': [
                            "Antimalarial medication immediately",
                            "Fever management with antipyretics",
                            "Stay well hydrated",
                            "Blood tests for confirmation"
                        ]
                    }
                    
                    if disease in treatment_map:
                        cols_rec = st.columns(2)
                        with cols_rec[0]:
                            st.markdown("**Immediate Care:**")
                            for rec in treatment_map[disease][:2]:
                                st.markdown(f"✅ {rec}")
                        
                        with cols_rec[1]:
                            st.markdown("**Follow-up Care:**")
                            for rec in treatment_map[disease][2:]:
                                st.markdown(f"📅 {rec}")
                    else:
                        st.info("""
                        **General Recommendations:**
                        - Monitor symptoms closely
                        - Maintain proper hydration
                        - Get adequate rest
                        - Follow up with healthcare provider within 24-48 hours
                        - Keep a record of symptom progression
                        """)
                    
                    # Save to history
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        'Time': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        'Diagnosis': disease,
                        'Confidence': f"{confidence:.1f}%",
                        'Symptoms': current_symptoms[:50] + ("..." if len(current_symptoms) > 50 else "")
                    })
                    
                    # Keep only last 5 entries
                    if len(st.session_state.history) > 5:
                        st.session_state.history = st.session_state.history[-5:]
                
                else:
                    st.error("❌ Could not analyze symptoms. Please try different wording.")
    
    # Show history
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        with st.expander("📖 Recent Diagnoses (Last 5)"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            
            if st.button("Clear History", key="clear_history_btn"):
                st.session_state.history = []
                st.rerun()
    
    # Sample test cases
    st.markdown("---")
    st.markdown("### 🧪 Try Sample Cases")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    if sample_col1.button("Test: Skin Issues", key="sample1_btn"):
        st.session_state.symptom_input = "I have red itchy skin with silver scales on my elbows and knees"
        st.rerun()
    
    if sample_col2.button("Test: Headache", key="sample2_btn"):
        st.session_state.symptom_input = "Severe headache on one side with sensitivity to light and sound"
        st.rerun()
    
    if sample_col3.button("Test: Fever", key="sample3_btn"):
        st.session_state.symptom_input = "High fever with chills and body pain for several days"
        st.rerun()
    
    # Model info
    with st.expander("ℹ️ About the AI Model"):
        st.markdown(f"""
        **Model Information:**
        - **Algorithm:** Random Forest Classifier
        - **Number of Diseases:** {len(diseases)}
        - **Training Data:** Symptom2disease dataset
        - **Features:** 1000 TF-IDF text features
        
        **Sample Diseases Detected:**
        {', '.join(sorted(diseases)[:12])}
        
        **Performance:**
        - Typical Accuracy: 85-90%
        - Response Time: < 1 second
        - Offline Capable: Yes
        
        **Important Note:**
        This AI is a diagnostic **assistant tool only**.
        Always consult with qualified healthcare professionals for final diagnosis and treatment.
        """)

# ============================================
# PAGE 3: DATASET INFO
# ============================================
elif page == "Dataset Info":
    st.title("📊 Dataset Information")
    
    # Check if dataset exists
    dataset_exists = os.path.exists('Symptom2disease.csv') or os.path.exists('./Symptom2disease.csv')
    
    if not dataset_exists:
        st.error("""
        ❌ Dataset 'Symptom2disease.csv' not found in project folder!
        
        Please make sure:
        1. The CSV file is named exactly: `Symptom2disease.csv`
        2. It's in the same folder as `app.py`
        3. It contains symptom descriptions and disease labels
        """)
        st.stop()
    
    # Load dataset
    try:
        dataset_path = 'Symptom2disease.csv'
        if not os.path.exists(dataset_path):
            dataset_path = './Symptom2disease.csv'
        
        df = pd.read_csv(dataset_path)
        
        # Handle column names
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']].copy()
        elif 'symptom' in df.columns and 'disease' in df.columns:
            df = df.rename(columns={'symptom': 'text', 'disease': 'label'})
        elif len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['text', 'label']
        
        st.success(f"✅ Dataset loaded successfully: {len(df)} records")
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            unique_diseases = df['label'].nunique()
            st.metric("Unique Diseases", unique_diseases)
        
        with col3:
            avg_len = df['text'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_len:.0f} chars")
        
        with col4:
            common_disease = df['label'].mode().iloc[0] if not df.empty else "N/A"
            st.metric("Most Common", common_disease[:15])
        
        # Show data preview
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Disease distribution
        st.markdown("---")
        st.subheader("Disease Distribution (Top 20)")
        
        disease_counts = df['label'].value_counts().head(20).reset_index()
        disease_counts.columns = ['Disease', 'Count']
        
        import plotly.express as px
        fig = px.bar(disease_counts, x='Count', y='Disease', orientation='h',
                    color='Count', color_continuous_scale=['#00b4d8', '#0077b6'],
                    title='Most Common Diseases in Dataset')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample symptoms explorer
        st.markdown("---")
        st.subheader("Explore Symptom Descriptions")
        
        tab1, tab2 = st.tabs(["Browse by Disease", "Random Samples"])
        
        with tab1:
            selected_disease = st.selectbox("Select a disease to view symptoms:", 
                                           sorted(df['label'].unique()))
            
            disease_samples = df[df['label'] == selected_disease].head(10)
            st.write(f"**{len(disease_samples)} sample descriptions for {selected_disease}:**")
            
            for idx, row in disease_samples.iterrows():
                with st.expander(f"Symptom Example {idx+1}"):
                    st.write(row['text'])
        
        with tab2:
            random_samples = df.sample(min(10, len(df)))
            st.write("**Random symptom descriptions from dataset:**")
            
            for idx, row in random_samples.iterrows():
                with st.expander(f"{row['label']} - Sample {idx+1}"):
                    st.write(row['text'])
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ============================================
# PAGE 4: TRAIN MODEL
# ============================================
elif page == "Train Model":
    st.title("⚙️ Model Training")
    
    st.markdown("""
    ## Train Your AI Disease Diagnosis Model
    
    This page trains the AI model using your Symptom2disease dataset.
    """)
    
    # Check if dataset exists
    dataset_exists = os.path.exists('Symptom2disease.csv') or os.path.exists('./Symptom2disease.csv')
    
    if not dataset_exists:
        st.error("❌ Dataset 'Symptom2disease.csv' not found!")
        st.stop()
    
    # Current model status
    st.markdown("---")
    st.subheader("Current Status")
    
    model_exists = os.path.exists('models/trained_model.pkl') or os.path.exists('./models/trained_model.pkl')
    
    if model_exists:
        st.success("✅ Model is already trained!")
        
        try:
            model_path = 'models/trained_model.pkl'
            if not os.path.exists(model_path):
                model_path = './models/trained_model.pkl'
            
            model_data = joblib.load(model_path)
            diseases = model_data['label_encoder'].classes_
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.metric("Diseases Detected", len(diseases))
                st.metric("Model Type", "Random Forest")
            
            with col_stat2:
                st.markdown("**Sample Diseases:**")
                for disease in diseases[:6]:
                    st.write(f"- {disease}")
                if len(diseases) > 6:
                    st.write(f"- ... and {len(diseases)-6} more")
        
        except Exception as e:
            st.info("Model file exists but could not load details.")
    else:
        st.warning("⚠️ Model not trained yet")
        st.info("Click 'Start Training' button below to begin.")
    
    # Training section
    st.markdown("---")
    st.subheader("Run Training")
    
    # Create training button
    if st.button("▶️ START TRAINING", type="primary", use_container_width=True, key="train_main_btn"):
        # Show training output section
        st.markdown("---")
        st.subheader("Training Output")
        
        # Create output container
        output_container = st.empty()
        output_container.info("🔄 Starting training process... Please wait.")
        
        try:
            # Import training module
            from train_model import train_model
            
            # Run training
            with st.spinner("Training in progress..."):
                output = train_model()
            
            # Display output
            output_container.code(output, language='text')
            
            # Check if model was created
            model_created = os.path.exists('models/trained_model.pkl') or os.path.exists('./models/trained_model.pkl')
            
            if model_created:
                st.success("✅ Training completed successfully!")
                st.balloons()
                
                if st.button("🔄 Refresh Page", key="refresh_btn"):
                    st.rerun()
            else:
                st.error("Training completed but model file was not created.")
                
        except Exception as e:
            output_container.error(f"❌ Training failed: {str(e)}")
            st.error("Training failed due to an error.")
    
    st.markdown("---")
    with st.expander("📋 Training Instructions"):
        st.markdown("""
        **Already trained successfully in terminal?** 
        - Accuracy: 87.92%
        - Model saved to: models/trained_model.pkl
        
        **If training doesn't work above:**
        
        1. **Open Command Prompt/Terminal** in your project folder
        
        2. **Run this command:**
        ```bash
        python train_model.py
        ```
        
        3. **Wait for completion**
        
        4. **Refresh this page**
        """)
        
        # Quick check button
        if st.button("Check Model Files", key="check_files_btn"):
            col_check1, col_check2 = st.columns(2)
            
            with col_check1:
                model_exists = os.path.exists('models/trained_model.pkl') or os.path.exists('./models/trained_model.pkl')
                if model_exists:
                    st.success("✅ trained_model.pkl exists")
                    try:
                        if os.path.exists('models/trained_model.pkl'):
                            size = os.path.getsize('models/trained_model.pkl')
                        else:
                            size = os.path.getsize('./models/trained_model.pkl')
                        st.info(f"Size: {size:,} bytes")
                    except:
                        pass
                else:
                    st.error("❌ trained_model.pkl not found")
            
            with col_check2:
                if os.path.exists('models/diseases.csv') or os.path.exists('./models/diseases.csv'):
                    st.success("✅ diseases.csv exists")
                    try:
                        if os.path.exists('models/diseases.csv'):
                            diseases_df = pd.read_csv('models/diseases.csv')
                        else:
                            diseases_df = pd.read_csv('./models/diseases.csv')
                        st.info(f"Contains {len(diseases_df)} diseases")
                    except:
                        pass
                else:
                    st.error("❌ diseases.csv not found")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 AI Disease Diagnosis System | Research Project")