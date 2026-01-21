import streamlit as st
import os
import random
from utils.preprocessing import load_image, load_case_data, format_clinical_text

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Derm-Memory Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🩺 Rural Derm-AI")
    st.markdown("---")
    
    # Simulating a Doctor's Input for the Demo
    st.subheader("Demo Controls")
    
    # Allow user to pick a random case from your generated data to "Simulate" a patient visit
    available_cases = [f.replace(".json", "") for f in os.listdir("data/json") if f.endswith(".json")]
    selected_case_id = st.selectbox("Simulate Patient Arrival:", sorted(available_cases))
    
    st.info("In a real scenario, this data would come from the live camera/form input.")

# --- MAIN LAYOUT ---
st.title("Medical Case Memory Assistant")
st.markdown("### 🏥 Current Patient Visit")

# Load the selected simulated case
case_data = load_case_data(selected_case_id)

if case_data:
    # Create two columns: Left for Inputs, Right for AI Results
    col_input, col_results = st.columns([1, 1])

    # --- LEFT COLUMN: DOCTOR'S VIEW ---
    with col_input:
        st.subheader("📝 Clinical Observations")
        
        # 1. Image Display
        img_filename = case_data["files"]["image_main"]
        img = load_image(img_filename)
        
        if img:
            st.image(img, caption=f"Lesion Capture: {img_filename}", use_container_width=True)
        else:
            st.error(f"Image not found: {img_filename}")

        # 2. Clinical Notes (Editable)
        # We pre-fill this with your synthetic data to save time during the demo
        formatted_text = format_clinical_text(case_data)
        notes = st.text_area("Doctor's Notes (Voice/Text):", value=formatted_text, height=150)
        
        # 3. Search Button
        if st.button("🔍 Search Collective Memory", type="primary", use_container_width=True):
            st.session_state['searching'] = True

    # --- RIGHT COLUMN: AI MEMORY RESULTS ---
    with col_results:
        st.subheader("🧠 Similar Historical Cases")
        
        if st.session_state.get('searching'):
            # PLACEHOLDER: This is where Person A's search logic will go later
            # For now, we just show a static message to prove the UI works.
            
            with st.spinner("Analyzing visual morphology & querying 50+ PHC records..."):
                import time; time.sleep(1.5) # Fake loading for effect
                
            st.success(f"Found 3 similar cases for {case_data['demographics']['location']}")
            
            # Mock Result Card (We will make this real in Phase 4)
            st.markdown("---")
            st.markdown(f"**MATCH #1: (89% Similarity)**")
            st.info(f"Diagnosis: {case_data['diagnosis']} (Self-Match)") 
            st.caption("Outcome: " + case_data['outcome'])
            
else:
    st.warning("Please select a case from the sidebar to begin.")