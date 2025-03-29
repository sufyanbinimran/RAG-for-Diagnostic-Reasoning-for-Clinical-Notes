import streamlit as st
import pandas as pd
import faiss
import numpy as np
import requests
import asyncio
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Hugging Face API Details
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B"
HF_API_KEY = "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"  # Replace with your API key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    return pd.read_pickle("preprocessed_medical_data.pkl")

medical_df = load_data()

# ✅ Load & Cache Embedding Model
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

# ✅ Retrieve Relevant Medical Data
async def retrieve_documents(query, top_n=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)
    _, faiss_top_n = faiss_index.search(query_embedding, top_n)
    return medical_df.iloc[list(faiss_top_n[0])][['diagnosis', 'combined_text']]

# ✅ Generate Doctor’s Report
async def generate_doctor_report(answers, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])

    prompt = f"""
You are an AI doctor. Based on the following patient responses and retrieved medical records, generate a structured medical report.

=== Patient Responses ===
{answers}

=== Retrieved Medical Records ===
{truncated_text}

Generate the report in this format:

================ Doctor’s Report ================
✅ Chief Complaint: [Summarize main symptoms]
✅ Medical History: [Summarize chronic conditions, past illnesses]
✅ Examination Findings: [Summarize vitals, observations]
✅ Possible Diagnoses: [List potential causes]
✅ Recommended Tests: [Suggest diagnostic tests]
✅ Treatment Plan: [Provide treatment recommendations]
================================================
"""

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
        timeout=30
    )

    return response.json()[0]["generated_text"] if response.status_code == 200 else "⚠️ Error generating report."

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")

st.header("🔹 Questions a Doctor Asks")
general = st.text_input("✅ General: What brings you here? How long have you had symptoms?")
symptoms = st.text_area("✅ Symptoms: Describe symptoms. Pain level (1-10)? Any patterns?")
medical_history = st.text_area("✅ Medical History: Any chronic conditions, past surgeries, medications?")
family_history = st.text_area("✅ Family History: Any genetic disorders, heart disease, cancer in family?")
lifestyle = st.text_area("✅ Lifestyle: Do you smoke, drink, exercise? Sleep quality?")
specific = st.text_area("✅ Specific (Based on Symptoms): Fever (recent travel?), Cough (shortness of breath?), Pain (location, triggers?)")

if st.button("Generate Doctor’s Report"):
    user_answers = f"General: {general}\nSymptoms: {symptoms}\nMedical History: {medical_history}\nFamily History: {family_history}\nLifestyle: {lifestyle}\nSpecific: {specific}"

    if symptoms.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = asyncio.run(retrieve_documents(symptoms))

        with st.spinner("🧠 Generating Doctor’s Report..."):
            report = asyncio.run(generate_doctor_report(user_answers, retrieved_results))

        st.header("🔹 Doctor’s Report")
        st.text(report)
    else:
        st.warning("⚠️ Please enter symptoms to proceed.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
