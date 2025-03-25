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
HF_API_KEY = "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"  # Replace with your actual API key
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
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index()

# ✅ Hybrid Retrieval Function (Async)
async def retrieve_documents(query, top_n=3):
    query_tokens = query.lower().split()
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Hugging Face API Text Generation
async def generate_medical_summary(user_query, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])

    prompt = f"""
You are a medical AI assistant providing structured reports based on medical records.

**User Query:** {user_query}

**Retrieved Medical Records:** {truncated_text}

Generate the report in this format:

**Diagnosis:** Summarize the diagnosis.
**Symptoms:** List key symptoms.
**Medical Details:** Summarize tests, results, findings, history.
**Treatment & Cure:** Mention treatments suggested.
**Physical Examination Findings:** Summarize physical examination.

Provide the final structured report:
    """

    for attempt in range(3):
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
                return "⚠️ Input too long. Try a shorter query."
            else:
                return f"⚠️ Error {response.status_code}: {response.json()}"

        except requests.exceptions.RequestException as e:
            if attempt < 2:
                st.warning(f"Retrying due to network error... ({attempt + 1}/3)")
            else:
                return "⚠️ API request failed after multiple attempts. Try again later."

# ✅ Add Beautiful Custom Styling
st.markdown("""
    <style>
    .report-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    .section-title {
        color: #004d99;
        font-size: 22px;
        margin-bottom: 10px;
    }
    .report-text {
        font-size: 16px;
        line-height: 1.6;
        color: #333333;
    }
    .header {
        color: #2e86de;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        color: #777;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Main App Title
st.markdown('<div class="header">🩺 AI-Powered Medical Report Generator</div>', unsafe_allow_html=True)
st.write("Provide a medical case or patient symptoms to generate a detailed, structured report.")

# ✅ User Query Input
query = st.text_area("🔍 **Enter Medical Query:**", placeholder="E.g., Diabetic patient with foot pain and numbness", height=150)

if st.button("🚀 Generate Report"):
    if query.strip():
        with st.spinner("🔄 Retrieving relevant medical records..."):
            retrieved_results = asyncio.run(retrieve_documents(query))

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating structured medical report..."):
                summary = asyncio.run(generate_medical_summary(query, retrieved_results))

            # ✅ Display Generated Report with Beautiful Formatting
            st.markdown("### 📄 Generated Medical Report")
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.markdown(f"<div class='report-text'>{summary.replace('**', '<b>').replace('__', '</b>')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ✅ Optional: Download Report
            st.download_button(
                label="💾 Download Report",
                data=summary,
                file_name="medical_report.txt",
                mime="text/plain"
            )

        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Footer
st.markdown('<div class="footer">Developed with ❤️ for Medical Professionals</div>', unsafe_allow_html=True)

# ✅ Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
