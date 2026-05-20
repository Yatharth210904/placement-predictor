import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the saved AI files safely
try:
    clf_model = joblib.load('placement_classifier.pkl')
    reg_model = joblib.load('salary_regressor.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    le_stream = joblib.load('stream_encoder.pkl')
    le_work_exp = joblib.load('work_exp_encoder.pkl')
except:
    st.error("Please run your training script first to generate the basic AI models!")

st.set_page_config(page_title="Placement Hub", layout="centered")
st.title("🎓 Smart Campus Placement Hub")

# Create two clean navigation tabs
tab1, tab2 = st.tabs(["🔮 Predict Your Status", "📥 Contribute Real Data"])

# ==========================================
# TAB 1: THE PREDICTION ENGINE
# ==========================================
with tab1:
    st.write("Enter your statistics to calculate your placement probability.")
    
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, step=0.1)
        aptitude = st.slider("Aptitude Test Score (%)", 0, 100, 70)
        internships = st.selectbox("Internships Completed", [0, 1, 2, 3, 4], key="pred_intern")
    with col2:
        backlogs = st.selectbox("Active Backlogs", [0, 1, 2, 3], key="pred_back")
        stream = st.selectbox("Engineering Stream", ["CS", "IT", "ECE", "EE"], key="pred_stream")
        work_exp = st.selectbox("Prior Work Experience?", ["No", "Yes"], key="pred_exp")

    if st.button("Predict Placement Status", type="primary"):
        stream_encoded = le_stream.transform([stream])[0]
        work_exp_encoded = le_work_exp.transform([work_exp])[0]
        
        user_data = pd.DataFrame([[cgpa, aptitude, internships, backlogs, stream_encoded, work_exp_encoded]],
                                 columns=['cgpa', 'aptitude_score', 'internships', 'backlogs', 'stream', 'work_exp'])
        
        user_data_scaled = scaler.transform(user_data)
        prediction = clf_model.predict(user_data_scaled)[0]
        
        st.markdown("---")
        if prediction == "Placed":
            predicted_salary = reg_model.predict(user_data_scaled)[0]
            st.success(f"🎉 Prediction: **PLACED**")
            st.metric(label="Estimated Salary Package", value=f"{predicted_salary:.2f} LPA")
        else:
            st.error("❌ Prediction: **NOT PLACED YET**")

# ==========================================
# TAB 2: REAL-WORLD DATA COLLECTION FORM
# ==========================================
with tab2:
    st.write("### Help Improve the AI: Submit Your Actual Placement Results")
    st.write("If you are a senior or a recently placed student, enter your true stats below to retrain our machine learning models.")
    
    with st.form("data_collection_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            user_cgpa = st.number_input("Final Graduation CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
            user_aptitude = st.slider("Average Aptitude Score (%)", 0, 100, 70, key="col_apt")
            user_internships = st.selectbox("Total Internships Completed", [0, 1, 2, 3, 4], key="col_intern")
        with c2:
            user_backlogs = st.selectbox("Total Backlogs during placements", [0, 1, 2, 3], key="col_back")
            user_stream = st.selectbox("Branch / Stream", ["CS", "IT", "ECE", "EE"], key="col_stream")
            user_work_exp = st.selectbox("Had Prior Work Experience?", ["No", "Yes"], key="col_exp")
        
        st.markdown("##### Actual Placement Outcome")
        actual_status = st.radio("Were you actually placed through campus drives?", ["Placed", "Not Placed"])
        
        actual_salary = 0.0
        if actual_status == "Placed":
            actual_salary = st.number_input("Actual Salary Package Offered (in LPA)", min_value=0.0, value=4.5, step=0.1)

        submit_data = st.form_submit_button("Submit Anonymous Data to AI", type="secondary")
        
        if submit_data:
            # Structuring the real human data entry row
            real_entry = {
                'cgpa': user_cgpa,
                'aptitude_score': user_aptitude,
                'internships': user_internships,
                'backlogs': user_backlogs,
                'stream': user_stream,
                'work_exp': user_work_exp,
                'status': actual_status,
                'salary': round(actual_salary, 2)
            }
            
            csv_filename = 'collected_user_data.csv'
            new_df = pd.DataFrame([real_entry])
            
            # Save smoothly to your laptop storage file
            if not os.path.isfile(csv_filename):
                new_df.to_csv(csv_filename, index=False)
            else:
                new_df.to_csv(csv_filename, mode='a', header=False, index=False)
                
            st.success("Thank you! Your verified entry has been safely logged to our training base.")
