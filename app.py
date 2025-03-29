import streamlit as st
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Medical AI Assistant", layout="wide")

@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df

medical_df = load_data()

@st.cache_data
def init_bm25():
    bm25_corpus = [text.split() for text in medical_df['combined_text']]
    return BM25Okapi(bm25_corpus)

bm25 = init_bm25()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_resource
def build_faiss_index():
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index()

@st.cache_resource
def load_local_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_local_model()

def retrieve_documents(query, top_n=3):
    query_tokens = query.lower().split()
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

def generate_medical_summary(user_query, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])

    prompt = f"""
You are a professional medical AI assistant. Based on the following patient data, generate a structured medical report.

=== User Query ===
{user_query}

=== Retrieved Medical Records ===
{truncated_text}

Format the output as:
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

st.title("🩺 Medical AI Assistant")
st.write("Enter medical case details to generate a structured medical report.")

query = st.text_area("🔍 Enter Medical Query:", placeholder="E.g., Diabetic patient with foot pain and numbness")

if st.button("Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = retrieve_documents(query)

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                summary = generate_medical_summary(query, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            st.text(summary)
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
