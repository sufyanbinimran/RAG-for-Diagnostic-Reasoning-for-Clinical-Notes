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

# ✅ Hugging Face API Details
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"  # Change if using a different model
HF_API_KEY = "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"  # Replace with your API key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    diagnosis_df = pd.read_pickle("preprocessed_diagnosis_data.pkl")

    # Combine relevant fields for retrieval
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df, diagnosis_df

medical_df, diagnosis_df = load_data()

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

# ✅ Hybrid Retrieval Function
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

# ✅ Generate Structured Medical Report
async def generate_medical_summary(user_query, retrieved_docs):
    # ✅ Extract Structured Information from Retrieved Docs
    diagnosis = retrieved_docs['diagnosis'].dropna().tolist()
    combined_text = " ".join(retrieved_docs['combined_text'].dropna().tolist())

    # ✅ Truncate Retrieved Text to Prevent Token Overflow
    truncated_text = " ".join(combined_text.split()[:500])  

    # ✅ Prepare Prompt for LLM
    prompt = f"""
    You are a medical AI assistant providing structured reports based on retrieved medical records.
    Given the user query and extracted medical records, generate a professional, structured report.

    **User Query:** {user_query}

    **Retrieved Information:**
    - **Diagnosis:** {', '.join(diagnosis) if diagnosis else "N/A"}
    - **Symptoms:** (Extract from retrieved records)
    - **Medical Details:** (Extract from retrieved records)
    - **Treatment & Cure:** (Infer based on retrieved medical knowledge)
    - **Physical Examination Findings:** (Extract if available)

    **Complete Structured Report:**
    - **Diagnosis:** {', '.join(diagnosis) if diagnosis else "N/A"}
    - **Symptoms:** (Extract explicitly from records)
    - **Medical Details:** (Extract relevant knowledge)
    - **Treatment & Cure:** (Provide best treatment approach)
    - **Physical Examination Findings:** (Extract if available)

    If any information is missing, use your medical reasoning to infer plausible details.
    Ensure the report is **concise, professional, and informative**.
    """

    # ✅ Call Hugging Face API
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 350}},
                timeout=30
            )

            if response.status_code == 200:
                json_response = response.json()
                if isinstance(json_response, list) and "generated_text" in json_response[0]:
                    return json_response[0]["generated_text"]
                else:
                    return "⚠️ API returned an unexpected response format."

            elif response.status_code == 422:
                return "⚠️ Input too long. Try a shorter query."

            else:
                return f"⚠️ Error {response.status_code}: {response.json()}"

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network error: {e}")
            if attempt < max_retries - 1:
                st.warning(f"Retrying... ({attempt+1}/{max_retries})")
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

            st.subheader("📄 Generated Medical Report:")
            st.markdown(f"```{summary}```")
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
