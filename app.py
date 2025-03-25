# ✅ Import Necessary Libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import torch
import nest_asyncio
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Apply AsyncIO Fix
nest_asyncio.apply()

# ✅ Set Streamlit Page Configuration (Must be First)
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Load Medical LLM (BioGPT-Large)
model_name = "microsoft/BioGPT-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Load Preprocessed Data
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    diagnosis_df = pd.read_pickle("preprocessed_diagnosis_data.pkl")
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df, diagnosis_df

medical_df, diagnosis_df = load_data()

# ✅ Tokenize for BM25
bm25_corpus = [text.split() for text in medical_df['combined_text']]
bm25 = BM25Okapi(bm25_corpus)

# ✅ Load Dense Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Convert Text Data to Embeddings
embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])

# ✅ FAISS Index for Efficient Dense Retrieval
d = embeddings.shape[1]  # Embedding dimension
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(embeddings)

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
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Generate Structured Medical Report
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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = generator.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.write("Enter a medical case or symptoms to generate a structured medical report.")

query = st.text_area("🔍 Enter Medical Query:", placeholder="E.g., Diabetic patient with foot pain and numbness")

if st.button("Generate Report"):
    if query.strip():
        retrieved_results = retrieve_documents(query)
        if not retrieved_results.empty:
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
