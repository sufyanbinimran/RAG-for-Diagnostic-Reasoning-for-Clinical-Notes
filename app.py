# ✅ Import Necessary Libraries
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

# ✅ Hugging Face API Details (Using Falcon-7B-Instruct)
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_API_KEY = "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"  # Replace with your API key
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

# ✅ Hugging Face API-Based Text Generation
async def generate_medical_summary(user_query, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    # ✅ Refined Prompt for Professional Summary
    prompt = f"""
You are a professional medical AI assistant. Based on the following patient data and medical records, generate a clean, well-structured medical report.

=== User Query ===
{user_query}

=== Retrieved Medical Records ===
{truncated_text}

Generate the report **strictly** in the following format (Use bullet points where necessary):

================ Medical Report ================
**Diagnosis:** 
- [List the diagnosis from the records]

**Symptoms:** 
- [List key symptoms]

**Medical Details:** 
- [Summarize relevant tests, findings, and medical history]

**Treatment & Cure:** 
- [Mention any treatment or suggested plan]

**Physical Examination Findings:** 
- [Summarize any physical examinations or vital signs]

Ensure the report is easy to read, medically professional, and structured with bullet points for each section.
================================================
"""

    # ✅ Retry API Call if it Fails
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
                timeout=30
            )

            if response.status_code == 200:
                json_response = response.json()
                if isinstance(json_response, list) and "generated_text" in json_response[0]:
                    return json_response[0]["generated_text"]
                else:
                    return "⚠️ API returned an unexpected response format."
            elif response.status_code == 422:
                return "⚠️ Input too long. Please try a shorter query."
            else:
                return f"⚠️ Error {response.status_code}: {response.json()}"

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network error: {e}")
            if attempt < max_retries - 1:
                st.warning(f"Retrying... ({attempt + 1}/{max_retries})")
            else:
                return "⚠️ API request failed after multiple attempts. Please try again later."

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

            # ✅ FINAL OUTPUT - ONLY Summary Displayed
            st.subheader("📄 Generated Medical Report:")
            st.text(summary)   # Displays ONLY the clean summary
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
