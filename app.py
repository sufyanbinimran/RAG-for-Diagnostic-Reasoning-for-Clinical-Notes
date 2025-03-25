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

# ✅ Fix AsyncIO Issues
nest_asyncio.apply()

# ✅ Streamlit Page Configuration (First Command)
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Load Medical LLM (BioGPT-Large)
@st.cache_resource
def load_model():
    model_name = "microsoft/BioGPT-Large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, generator

tokenizer, generator = load_model()

# ✅ Load Preprocessed Medical Data
@st.cache_data
def load_data():
    try:
        medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
        medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
        return medical_df
    except Exception as e:
        st.error(f"Error loading medical data: {e}")
        return None

medical_df = load_data()
if medical_df is None:
    st.stop()  # Stop execution if data loading fails

# ✅ Tokenize for BM25
bm25_corpus = [text.split() for text in medical_df['combined_text']]
bm25 = BM25Okapi(bm25_corpus)

# ✅ Load Sentence Transformer for Dense Retrieval
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ✅ Convert Text Data to Embeddings & Build FAISS Index
@st.cache_resource
def build_faiss_index():
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])
    d = embeddings.shape[1]  # Embedding dimension
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    return faiss_index, embeddings

faiss_index, embeddings = build_faiss_index()

# ✅ Hybrid Retrieval Function
def retrieve_documents(query, top_n=5):
    if not query:
        return pd.DataFrame()  # Return empty if query is blank

    # ✅ BM25 Retrieval
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    # ✅ FAISS Dense Retrieval
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)
    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    # ✅ Combine BM25 & FAISS Results
    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Generate Structured Medical Report
def generate_medical_summary(user_query, retrieved_docs):
    if retrieved_docs.empty:
        return "⚠️ No relevant medical records found."

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
st.write("Enter a **medical case or symptoms** to generate a structured **medical report.**")

query = st.text_area("🔍 **Enter Medical Query:**", placeholder="E.g., Diabetic patient with foot pain and numbness")

if st.button("Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = retrieve_documents(query)

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating medical summary..."):
                summary = generate_medical_summary(query, retrieved_results)

            st.subheader("📄 **Generated Medical Report:**")
            st.write(summary)
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Footer
st.markdown("""
---
💡 **Developed by Muhammad Sufyan Malik** | 🚀 **AI-Powered Clinical Assistant**
""")
