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
    
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]
    
    _, faiss_top_n = faiss_index.search(query_embedding, top_n)
    
    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]
    
    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Generate Medical Report
def generate_medical_summary(responses, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words
    
    prompt = f"""
You are a professional medical AI assistant. Based on the following patient data, generate a structured medical report.

=== Patient Responses ===
{responses}

=== Retrieved Medical Records ===
{truncated_text}

🔹 Doctor’s Report
✅ Chief Complaint: 
✅ Medical History: 
✅ Examination Findings: 
✅ Possible Diagnoses: 
✅ Recommended Tests: 
✅ Treatment Plan: 
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=500, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Answer the following questions to generate a structured medical report.")

responses = {}
responses["General"] = st.text_input("✅ What brings you here? How long have you had symptoms?")
responses["Symptoms"] = st.text_area("✅ Describe your symptoms. Pain level (1-10)? Any patterns?")
responses["Medical History"] = st.text_area("✅ Any chronic conditions, past surgeries, medications?")
responses["Family History"] = st.text_area("✅ Any genetic disorders, heart disease, or cancer in family?")
responses["Lifestyle"] = st.text_area("✅ Do you smoke, drink, exercise? Sleep quality?")
responses["Specific"] = st.text_area("✅ Specific Symptoms: Fever (recent travel?), Cough (shortness of breath?), Pain (location, triggers?)")

if st.button("Generate Report"):
    if any(response.strip() for response in responses.values()):
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = retrieve_documents(" ".join(responses.values()))

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                summary = generate_medical_summary(responses, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            st.text(summary)
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your responses.")
    else:
        st.error("❌ Please answer at least one question before generating a report.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
