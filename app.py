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
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_API_KEY = "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"  # Replace with actual API key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    med_data = pd.read_pickle("preprocessed_medical_data.pkl")
    diag_data = pd.read_pickle("preprocessed_diagnosis_data.pkl")
    
    # Ensure text is combined for retrieval
    med_data['combined_text'] = med_data[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    
    return med_data, diag_data

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
    d = embeddings.shape[1]
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

    # ✅ Extract Required Fields
    extracted_info = {
        "Diagnosis": retrieved_data["diagnosis"].tolist(),
        "Symptoms": retrieved_data["symptoms"].tolist(),
        "Medical Details": retrieved_data["combined_text"].tolist(),
        "Treatment & Cure": retrieved_data["treatment"].tolist(),
        "Physical Examination Findings": retrieved_data["physical_examination"].tolist()
    }

    return extracted_info

# ✅ Improved LLM Prompt & Generation
async def generate_medical_summary(user_query, retrieved_info):
    # ✅ Format Retrieved Information
    formatted_info = "\n".join([f"**{key}:** {', '.join(val)}" for key, val in retrieved_info.items() if val])

    prompt = f"""
    You are a medical AI assistant providing structured medical reports.
    
    **User Query:** {user_query}
    
    **Retrieved Medical Records:**
    {formatted_info}
    
    **Structured Medical Report:**
    - **Diagnosis:** Extracted or inferred from retrieved records.
    - **Symptoms:** Extracted from retrieved records.
    - **Medical Details:** Summarized from retrieved data.
    - **Treatment & Cure:** Provide recommended treatment based on the extracted medical details.
    - **Physical Examination Findings:** If available, extract from retrieved data.
    
    If any section is missing, generate it based on medical knowledge.
    """

    # ✅ Send Request to Hugging Face API
    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 250}},  # Fixed token limit
        timeout=30
    )

    # ✅ Handle API Response
    if response.status_code == 200:
        json_response = response.json()
        return json_response[0].get("generated_text", "⚠️ No valid response from LLM.")
    else:
        return f"⚠️ API Error {response.status_code}: {response.json()}"

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Enter a medical case or symptoms to generate a structured medical report.")

query = st.text_area("🔍 Enter Medical Query:", placeholder="E.g., Patient with fever and cough")

if st.button("Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = asyncio.run(retrieve_documents(query))

        if retrieved_results and any(retrieved_results.values()):
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
