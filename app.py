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

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
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

# ✅ Generate Summary using Local Model
def generate_medical_summary(user_inputs, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    # ✅ Structured prompt with doctor-like responses
    prompt = f"""
    You are a professional medical AI assistant. Based on the following patient details, generate a structured medical report.

    === Patient Information ===
    Chief Complaint: {user_inputs['chief_complaint']}
    Symptoms: {user_inputs['symptoms']}
    Pain Level: {user_inputs['pain_level']}
    Chronic Conditions: {user_inputs['chronic_conditions']}
    Medications: {user_inputs['medications']}
    Family History: {user_inputs['family_history']}
    Lifestyle: {user_inputs['lifestyle']}
    Specific Symptoms: {user_inputs['specific_symptoms']}

    === Retrieved Medical Records ===
    {truncated_text}

    Format the output as:
    Chief Complaint: 
    - [Details]
    
    Medical History: 
    - [Details]
    
    Examination Findings: 
    - [Details]
    
    Possible Diagnoses: 
    - [Details]
    
    Recommended Tests: 
    - [Details]
    
    Treatment Plan:
    - [Details]
    """

    # ✅ Tokenize & Generate Response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=500, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Answer the following questions to generate a structured medical report.")

# 🔹 Collect Patient Information
chief_complaint = st.text_input("🔹 Chief Complaint:", placeholder="E.g., Persistent cough for 5 days")
symptoms = st.text_area("🔹 Describe Symptoms:", placeholder="E.g., Fever, chills, body aches")
pain_level = st.slider("🔹 Pain Level (1-10):", 1, 10, 5)
chronic_conditions = st.text_input("🔹 Chronic Conditions:", placeholder="E.g., Diabetes, Hypertension")
medications = st.text_input("🔹 Current Medications:", placeholder="E.g., Metformin, Lisinopril")
family_history = st.text_area("🔹 Family History:", placeholder="E.g., Heart disease, Diabetes in parents")
lifestyle = st.text_area("🔹 Lifestyle (Smoking, Alcohol, Exercise):", placeholder="E.g., Non-smoker, drinks occasionally")
specific_symptoms = st.text_area("🔹 Specific Symptoms:", placeholder="E.g., Fever with recent travel history")

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
if st.button("Generate Medical Report"):
    if any(value.strip() for value in user_inputs.values()):
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = retrieve_documents(user_inputs["chief_complaint"])

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                summary = generate_medical_summary(user_inputs, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            st.markdown(f"```{summary}```")

            # ✅ Add Download Button
            report_filename = "medical_report.txt"
            st.download_button(
                label="⬇️ Download Report",
                data=summary,
                file_name=report_filename,
                mime="text/plain"
            )
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your input.")
    else:
        st.error("❌ Please fill out at least one field.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
