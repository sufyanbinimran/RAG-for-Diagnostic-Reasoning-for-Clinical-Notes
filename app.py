# ✅ Import Necessary Libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import requests
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ Hugging Face API Details (Using Falcon-7B-Instruct)
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_API_KEY = "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"  # 🔹 Replace with your API key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Load & Cache Preprocessed Medical Data
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    diagnosis_df = pd.read_pickle("preprocessed_diagnosis_data.pkl")

    # ✅ Fill missing values
    medical_df.fillna("N/A", inplace=True)
    diagnosis_df.fillna("N/A", inplace=True)

    return medical_df, diagnosis_df

medical_df, diagnosis_df = load_data()

# ✅ Combine Available Information for Retrieval
medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
diagnosis_df['combined_text'] = diagnosis_df.astype(str).agg(' '.join, axis=1)

# ✅ Tokenize for BM25 (Cached)
@st.cache_data
def init_bm25():
    combined_corpus = [text.split() for text in medical_df['combined_text']] + \
                      [text.split() for text in diagnosis_df['combined_text']]
    return BM25Okapi(combined_corpus)

bm25 = init_bm25()

# ✅ Load & Cache Dense Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ✅ Compute & Cache FAISS Index
@st.cache_resource
def build_faiss_index():
    combined_texts = list(medical_df['combined_text']) + list(diagnosis_df['combined_text'])
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in combined_texts])

    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index

faiss_index = build_faiss_index()

# ✅ Hybrid Retrieval Function
def retrieve_documents(query, top_n=5):
    query_tokens = query.lower().split()
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    # ✅ BM25 Retrieval
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    # ✅ FAISS Dense Retrieval
    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    # ✅ Combine Results
    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])

    # ✅ Filter Out-of-Bounds Indices
    valid_retrieved_docs_medical = [i for i in retrieved_docs if i < len(medical_df)]
    valid_retrieved_docs_diagnosis = [i for i in retrieved_docs if i < len(diagnosis_df)]

    # ✅ Extracting Information from Both DataFrames
    retrieved_data_medical = medical_df.iloc[valid_retrieved_docs_medical, :]
    retrieved_data_diagnosis = diagnosis_df.iloc[valid_retrieved_docs_diagnosis, :]

    # ✅ Limit the number of records for the model input
    retrieved_data_medical = retrieved_data_medical.head(3)  # Limit to 3 records
    retrieved_data_diagnosis = retrieved_data_diagnosis.head(3)  # Limit to 3 records

    # ✅ Merge retrieved data
    retrieved_data = pd.concat([retrieved_data_medical, retrieved_data_diagnosis], axis=0)

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Generate Structured Medical Report via Hugging Face API
def generate_medical_summary(user_query, retrieved_docs):
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

    Generate a professional and well-structured report based on the retrieved information.
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.7, "do_sample": True}
    }

    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text']
    else:
        st.error(f"API error: {response.status_code} - {response.text}")
        return "⚠️ Error: Failed to generate medical report."

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
                summary = generate_medical_summary(query, retrieved_results)

            st.subheader("📄 Generated Medical Report:")
            st.write(summary)

        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
