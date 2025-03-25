# ✅ Import Necessary Libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import requests
import torch
import asyncio
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ Streamlit Page Configuration (Must be First)
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Hugging Face API (Faster than Local Model)
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/BioGPT"
HF_API_KEY = "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"  # 🔥 Replace with your API key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

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

# ✅ Hybrid Retrieval Function (Async for Speed)
async def retrieve_documents(query, top_n=3):
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

# ✅ Hugging Face API-Based Generation (Super Fast)
async def generate_medical_summary(user_query, retrieved_docs):
    prompt = f"""
    You are a medical AI assistant providing structured reports based on retrieved medical records.
    Given the following information, generate a structured summary.

    **User Query:** {user_query}

    **Retrieved Medical Records:**
    {retrieved_docs.to_string(index=False)}

    **Structured Report:**
    - **Diagnosis:** (Extract from retrieved records)
    - **Symptoms:** (Extract from combined_text)
    - **Medical Details:** (Extract relevant knowledge)
    - **Treatment & Cure:** (Infer based on medical details)
    - **Physical Examination Findings:** (If available, extract from records)

    Generate a professional and well-structured report.
    """

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.7}},
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "⚠️ Error generating response. Try again later."

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Enter a medical case or symptoms to generate a structured medical report.")

query = st.text_area("🔍 Enter Medical Query:", placeholder="E.g., Diabetic patient with foot pain and numbness")

if st.button("Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = asyncio.run(retrieve_documents(query))

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                summary = asyncio.run(generate_medical_summary(query, retrieved_results))

            st.subheader("📄 Generated Medical Report:")
            st.markdown(f"```{summary}```")
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
