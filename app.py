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

# ✅ Load Medical LLM (Switching to Smaller Model for Faster Generation)
model_name = "microsoft/BioGPT"  # ✅ Using smaller model for faster response
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Load Preprocessed Data (Cache to Speed Up)
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df

medical_df = load_data()

# ✅ Tokenize for BM25 (Cache to Avoid Repeated Processing)
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

# ✅ Compute & Cache Embeddings for FAISS
@st.cache_resource
def build_faiss_index():
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings

faiss_index, embeddings = build_faiss_index()

# ✅ Hybrid Retrieval Function (Optimized for Speed)
def retrieve_documents(query, top_n=3):  # ✅ Reduced to top 3 for speed
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

# ✅ Generate Structured Medical Report (Optimized for Speed)
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

    Generate a concise, professional, and well-structured report based on the retrieved information.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # ✅ Reduced to 512 for speed
    output = generator.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)  # ✅ Reduced to 200 tokens

    return tokenizer.decode(output[0], skip_special_tokens=True)

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
