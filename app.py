// app.js
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ Streamlit Page Config
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .report-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .section-header {
        color: #2c3e50;
        font-weight: bold;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    # Sample data structure - replace with your actual data loading
    data = {
        'diagnosis': ['Hypertension', 'Diabetes Mellitus', 'Bronchial Asthma', 'Migraine'],
        'combined_text': [
            "Patient presents with elevated blood pressure, headache, and dizziness. Risk factors include family history and high sodium diet.",
            "Patient reports increased thirst, frequent urination, and fatigue. HbA1c levels elevated.",
            "Wheezing, shortness of breath, and chest tightness reported. Symptoms worse at night.",
            "Recurrent unilateral headache with photophobia and nausea. No aura present."
        ]
    }
    medical_df = pd.DataFrame(data)
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df

medical_df = load_data()

# ✅ Tokenize for BM25 (Cached)
@st.cache_data
def init_bm25():
    bm25_corpus = [text.split() for text in medical_df['combined_text']]
    return BM25Okapi(bm25_corpus)

bm25 = init_bm25()

# ✅ Load & Cache Dense Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ✅ Compute & Cache FAISS Index
@st.cache_resource
def build_faiss_index():
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index()

# ✅ Load Local Hugging Face Model (BART)
@st.cache_resource
def load_local_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_local_model()

# ✅ Hybrid Retrieval Function
def retrieve_documents(query, top_n=3):
    query_tokens = query.lower().split()
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    # ✅ BM25 Retrieval
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    # ✅ FAISS Dense Retrieval
    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    # ✅ Combine Results
    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Generate Medical Report
def generate_medical_report(user_inputs, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    # ✅ Doctor's Report Structure with enhanced prompt
    prompt = f"""
    Generate a comprehensive medical report based on the patient's details and relevant medical records.
    Use professional medical terminology and maintain a clinical tone throughout.
    
    === Patient Details ===
    Chief Complaint: {user_inputs['chief_complaint']}
    Symptoms: {user_inputs['symptoms']}
    Pain Level: {user_inputs['pain_level']}/10
    Chronic Conditions: {user_inputs['chronic_conditions']}
    Medications: {user_inputs['medications']}
    Family History: {user_inputs['family_history']}
    Lifestyle Factors: {user_inputs['lifestyle']}
    Specific Symptoms: {user_inputs['specific_symptoms']}

    === Relevant Medical Records ===
    {truncated_text}

    Generate a structured report with the following sections:
    
    Chief Complaint:
    - Clearly state the primary reason for the visit in 1-2 sentences
    
    Medical History:
    - Include relevant past medical history
    - Current medications and allergies
    - Family history of note
    - Lifestyle factors that may be relevant
    
    Examination Findings:
    - Document vital signs if available
    - Describe physical examination findings
    - Note any notable observations
    
    Possible Diagnoses:
    - List 2-3 most likely differential diagnoses
    - Briefly explain reasoning for each
    - Order by likelihood
    
    Recommended Tests:
    - Suggest appropriate diagnostic tests
    - Include both laboratory and imaging studies
    - Prioritize by clinical utility
    
    Treatment Plan:
    - Proposed immediate treatment
    - Medications with dosage if applicable
    - Lifestyle modifications
    - Follow-up recommendations
    """

    # ✅ Tokenize & Generate Response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=1000, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Complete the following form to generate a structured medical report.")

# 🔹 Collect Patient Information
with st.expander("Patient Information Form", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        chief_complaint = st.text_input("Chief Complaint*", placeholder="E.g., Chest pain for 2 hours")
        symptoms = st.text_area("Symptoms Description*", placeholder="Describe your symptoms in detail")
        pain_level = st.slider("Pain Level (1-10)", 1, 10, 3)
        chronic_conditions = st.text_input("Chronic Conditions", placeholder="E.g., Hypertension, Diabetes")
    
    with col2:
        medications = st.text_input("Current Medications", placeholder="Include dosage if known")
        family_history = st.text_area("Family History", placeholder="Relevant family medical history")
        lifestyle = st.text_area("Lifestyle Factors", placeholder="Smoking, alcohol, exercise habits")
        specific_symptoms = st.text_area("Specific Symptoms", placeholder="Any other notable symptoms")

# ✅ Store Responses
user_inputs = {
    "chief_complaint": chief_complaint,
    "symptoms": symptoms,
    "pain_level": pain_level,
    "chronic_conditions": chronic_conditions,
    "medications": medications,
    "family_history": family_history,
    "lifestyle": lifestyle,
    "specific_symptoms": specific_symptoms
}

# ✅ Generate Report
if st.button("Generate Medical Report", type="primary"):
    if chief_complaint.strip() and symptoms.strip():  # Require at least chief complaint and symptoms
        with st.spinner("🔄 Analyzing patient information and retrieving relevant medical knowledge..."):
            retrieved_results = retrieve_documents(f"{chief_complaint} {symptoms}")

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating comprehensive medical report..."):
                summary = generate_medical_report(user_inputs, retrieved_results)
            
            # Display the report with nice formatting
            st.subheader("Medical Report", divider="blue")
            
            # Parse the generated report into sections
            sections = {
                "Chief Complaint": "",
                "Medical History": "",
                "Examination Findings": "",
                "Possible Diagnoses": "",
                "Recommended Tests": "",
                "Treatment Plan": ""
            }
            
            current_section = None
            for line in summary.split('\n'):
                if ':' in line and line.split(':')[0].strip() in sections:
                    current_section = line.split(':')[0].strip()
                    sections[current_section] = line.split(':', 1)[1].strip()
                elif current_section:
                    sections[current_section] += '\n' + line.strip()
            
            # Display each section with custom styling
            for section, content in sections.items():
                if content.strip():
                    st.markdown(f"""
                    <div class="report-section">
                        <div class="section-header">{section}</div>
                        <div>{content}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ✅ Add Download Button
            report_filename = f"medical_report_{chief_complaint[:20].replace(' ', '_')}.txt"
            st.download_button(
                label="⬇️ Download Full Report",
                data=summary,
                file_name=report_filename,
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("No relevant medical records found. Please provide more detailed information.")
    else:
        st.error("Please fill at least the Chief Complaint and Symptoms fields (marked with *)")

if __name__ == "__main__":
    st.write("System ready for patient evaluation.")
