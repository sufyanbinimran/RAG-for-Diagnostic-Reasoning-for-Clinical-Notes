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

    # ✅ Check Available Columns Before Accessing
    available_columns = retrieved_data.columns.tolist()

    extracted_info = {
        "Diagnosis": retrieved_data["diagnosis"].tolist() if "diagnosis" in available_columns else ["Not Found"],
        "Symptoms": retrieved_data["symptoms"].tolist() if "symptoms" in available_columns else ["Not Found"],
        "Medical Details": retrieved_data["combined_text"].tolist() if "combined_text" in available_columns else ["Not Found"],
        "Treatment & Cure": retrieved_data["treatment"].tolist() if "treatment" in available_columns else ["Not Found"],
        "Physical Examination Findings": retrieved_data["physical_examination"].tolist() if "physical_examination" in available_columns else ["Not Found"]
    }

    return extracted_info

# ✅ Improved LLM Prompt & Generation
import time

async def generate_medical_summary(user_query, retrieved_docs):
    # ✅ Truncate retrieved records to avoid exceeding token limit
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    prompt = f"""
    You are a medical AI assistant providing structured reports based on retrieved medical records.
    Your task is to extract the most relevant medical details from the retrieved records and present them in a structured, professional format.

    **User Query:** {user_query}

    **Retrieved Medical Records:** {truncated_text}

    **Structured Medical Report:**
    - **Diagnosis:** Extracted from retrieved records.
    - **Symptoms:** Extracted from retrieved records.
    - **Medical Details:** Extracted from retrieved records.
    - **Treatment & Cure:** Extracted or inferred based on medical details.
    - **Physical Examination Findings:** Extracted from records if available.

    Ensure that the report is **accurate, professional, and well-structured**.
    """

    # ✅ Retry API Call if it Fails
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 250}},  # Ensure token limit
                timeout=60  # ⬆️ Increased timeout from default to 60 seconds
            )

            # ✅ If Response is Successful
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

        except requests.exceptions.ReadTimeout:
            st.warning(f"⚠️ Timeout occurred. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(5)  # ⏳ Wait before retrying

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network error: {e}")
            return "⚠️ API request failed. Please try again later."

    return "⚠️ API request timed out after multiple attempts. Please try again later."

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
