import streamlit as st
import pandas as pd
import faiss
import numpy as np
import requests
import asyncio
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ Streamlit Configuration
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_API_KEY = "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"  # Replace with your key
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Load & Cache Medical Data
@st.cache_data
def load_data():
    medical_df = pd.read_pickle("preprocessed_medical_data.pkl")
    medical_df['combined_text'] = medical_df[['diagnosis', 'combined_text']].astype(str).agg(' '.join, axis=1)
    return medical_df

medical_df = load_data()

# ✅ Tokenize for BM25
@st.cache_data
def init_bm25():
    bm25_corpus = [text.split() for text in medical_df['combined_text']]
    return BM25Okapi(bm25_corpus)

bm25 = init_bm25()

# ✅ Load Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ✅ FAISS Index
@st.cache_resource
def build_faiss_index():
    embeddings = np.array([embedding_model.encode(text, convert_to_tensor=False) for text in medical_df['combined_text']])
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index()

# ✅ Hybrid Retrieval
async def retrieve_documents(query, top_n=3):
    query_tokens = query.lower().split()
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:top_n]

    _, faiss_top_n = faiss_index.search(query_embedding, top_n)

    retrieved_docs = set(bm25_top_n) | set(faiss_top_n[0])
    retrieved_data = medical_df.iloc[list(retrieved_docs)]

    return retrieved_data[['diagnosis', 'combined_text']]

# ✅ Hugging Face API Call
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
                st.warning(f"Retrying... ({attempt + 1}/3)")
            else:
                return "⚠️ API request failed. Try again later."

# ✅ Custom Styling
st.markdown("""
    <style>
    .header { color: #2e86de; font-size: 36px; font-weight: bold; text-align: center; margin-bottom: 40px;}
    .section-title { color: #004d99; font-size: 24px; margin-bottom: 15px;}
    .query-box { background-color: #f0f8ff; padding: 20px; border-radius: 12px; margin-bottom: 30px;}
    .report-box { background-color: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    .report-text { font-size: 16px; line-height: 1.8; color: #333333;}
    .footer { text-align: center; color: #777; margin-top: 50px; font-size: 14px;}
    .download-btn {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# ✅ App Title
st.markdown('<div class="header">🩺 AI Medical Report Generator</div>', unsafe_allow_html=True)

st.markdown('<div class="query-box"><b>Provide patient symptoms or medical case details below:</b></div>', unsafe_allow_html=True)

# ✅ User Input
query = st.text_area("🔎 **Enter Medical Query:**", placeholder="E.g., Diabetic patient with foot pain and numbness", height=150)

if st.button("🚀 Generate Report"):
    if query.strip():
        with st.spinner("🔄 Fetching medical records..."):
            retrieved_results = asyncio.run(retrieve_documents(query))

        if not retrieved_results.empty:
            with st.spinner("🧠 Generating detailed report..."):
                summary = asyncio.run(generate_medical_summary(query, retrieved_results))

            # ✅ Beautiful Report Formatting
            formatted_summary = summary.replace("**Diagnosis:**", "<h3 style='color:#2980b9'>🩺 Diagnosis</h3>") \
                                       .replace("**Symptoms:**", "<h3 style='color:#2980b9'>🤒 Symptoms</h3>") \
                                       .replace("**Medical Details:**", "<h3 style='color:#2980b9'>📋 Medical Details</h3>") \
                                       .replace("**Treatment & Cure:**", "<h3 style='color:#2980b9'>💊 Treatment & Cure</h3>") \
                                       .replace("**Physical Examination Findings:**", "<h3 style='color:#2980b9'>🩻 Physical Examination Findings</h3>") \
                                       .replace("\n", "<br>")

            st.markdown('<div class="section-title">📄 Generated Medical Report</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-box"><div class="report-text">{formatted_summary}</div></div>', unsafe_allow_html=True)

            # ✅ Download Button
            st.download_button(
                label="💾 Download Report",
                data=summary,
                file_name="medical_report.txt",
                mime="text/plain",
                key="download_button"
            )
        else:
            st.warning("⚠️ No relevant medical records found. Please refine your query.")
    else:
        st.error("❌ Please enter a valid medical query.")

# ✅ Footer
st.markdown('<div class="footer">Developed with ❤️ for Medical Professionals | Powered by Falcon-7B</div>', unsafe_allow_html=True)

# ✅ Run
if __name__ == "__main__":
    st.write("🚀 AI Medical Assistant Ready!")
