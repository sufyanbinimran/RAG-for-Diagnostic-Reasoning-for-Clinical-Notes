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
    medical_df.fillna("N/A", inplace=True)  # Replace NaNs with "N/A" for missing values
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

    return retrieved_data

# ✅ Hugging Face API-Based Text Completion
def generate_missing_info(field_name, user_query, retrieved_text):
    prompt = f"""
    You are a medical AI assistant. Given a user query and retrieved medical records, generate the missing field: **{field_name}**.

    **User Query:** {user_query}
    **Retrieved Medical Records:** {retrieved_text}

    **{field_name}:** (Provide a well-structured response)
    """

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 100}},  
        timeout=30
    )

    if response.status_code == 200:
        json_response = response.json()
        if isinstance(json_response, list) and "generated_text" in json_response[0]:
            return json_response[0]["generated_text"]
    return "⚠️ Unable to generate this field."

# ✅ Extract & Complete Structured Report
def generate_structured_report(user_query, retrieved_docs):
    report = {}

    for field in ["Diagnosis", "Symptoms", "Medical Details", "Treatment & Cure", 
                  "Physical Examination Findings", "Patient Information", "History", 
                  "Physical Examination", "Diagnostic Tests", "Treatment & Management", 
                  "Follow-Up Care", "Outlook"]:
        
        retrieved_text = retrieved_docs[field.lower().replace(" ", "_")].to_string(index=False)
        
        if retrieved_text.strip() == "N/A":  # Use LLM only if data is missing
            report[field] = generate_missing_info(field, user_query, retrieved_docs.to_string(index=False))
        else:
            report[field] = retrieved_text

    return report

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Enter a medical case or symptoms to generate a structured medical report.")

query = st.text_area("🔍 Enter Medical Query:", placeholder="E.g., Diabetic patient with foot pain and numbness")

if st.button("Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = retrieve_documents(query)

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                report = generate_structured_report(query, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            for section, content in report.items():
                st.markdown(f"### {section}\n{content}")

        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
