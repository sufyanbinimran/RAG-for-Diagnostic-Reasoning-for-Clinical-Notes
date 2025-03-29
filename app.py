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
def generate_medical_summary(user_query, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    prompt = f"""
    You are a professional medical AI assistant. Based on the following patient data, generate a structured medical report.

    === Patient Query === {user_query}

    === Retrieved Medical Records === {truncated_text}

    Format the output as:
    ✅ Chief Complaint:
    ✅ Medical History:
    ✅ Examination Findings:
    ✅ Possible Diagnoses:
    ✅ Recommended Tests:
    ✅ Treatment Plan:
    """

    # ✅ Tokenize & Generate Response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=500, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# ✅ Streamlit UI - Medical AI Assistant
st.title("🩺 Medical AI Assistant")
st.write("Answer the following questions to generate a structured medical report.")

# ✅ Collect Patient Information
st.subheader("🔹 General Information")
chief_complaint = st.text_input("✅ What brings you here?")
duration = st.text_input("✅ How long have you had symptoms?")

st.subheader("🔹 Symptoms Details")
symptoms = st.text_area("✅ Describe your symptoms in detail.")
pain_level = st.slider("✅ Pain Level (1-10)?", 1, 10, 5)

st.subheader("🔹 Medical & Family History")
medical_history = st.text_area("✅ Any chronic conditions, past surgeries, or medications?")
family_history = st.text_area("✅ Any genetic disorders, heart disease, or cancer in your family?")

st.subheader("🔹 Lifestyle & Habits")
smoke_drink = st.radio("✅ Do you smoke or drink?", ("No", "Occasionally", "Regularly"))
exercise = st.radio("✅ Do you exercise?", ("Yes", "No"))
sleep_quality = st.slider("✅ How would you rate your sleep quality (1-10)?", 1, 10, 7)

st.subheader("🔹 Additional Symptoms (if applicable)")
fever = st.radio("✅ Do you have a fever?", ("No", "Yes, and I have traveled recently", "Yes, but no recent travel"))
cough = st.radio("✅ Do you have a cough?", ("No", "Yes, with shortness of breath", "Yes, but no breathing issues"))
pain_details = st.text_area("✅ If you have pain, where is it located and what triggers it?")

# ✅ Button to Generate Report
if st.button("Generate Medical Report"):
    if chief_complaint.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            query = f"{chief_complaint} {symptoms} {medical_history}"
            retrieved_results = retrieve_documents(query)

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                structured_input = f"""
                ✅ Chief Complaint: {chief_complaint}
                ✅ Duration: {duration}
                ✅ Symptoms: {symptoms}, Pain Level: {pain_level}
                ✅ Medical History: {medical_history}
                ✅ Family History: {family_history}
                ✅ Lifestyle: Smoking/Drinking: {smoke_drink}, Exercise: {exercise}, Sleep Quality: {sleep_quality}
                ✅ Additional Symptoms: Fever: {fever}, Cough: {cough}, Pain Details: {pain_details}
                """

                summary = generate_medical_summary(structured_input, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            st.markdown(f"```\n{summary}\n```")
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your responses.")
    else:
        st.error("❌ Please provide your chief complaint to generate the report.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
