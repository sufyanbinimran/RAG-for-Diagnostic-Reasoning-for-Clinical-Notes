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
st.set_page_config(
    page_title="Medical AI Assistant", 
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .report {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
    }
    .section-header {
        color: #2e59a9;
        font-weight: 600;
        margin-top: 15px;
    }
    .highlight {
        background-color: #f8f9fa;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 500;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Hugging Face API Details (Using Falcon-7B-Instruct)
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_API_KEY = "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"  # Replace with your API key
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
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index()

# ✅ Hybrid Retrieval Function (Async for Speed)
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

# ✅ Improved Hugging Face API-Based Text Generation
async def generate_medical_summary(user_query, retrieved_docs):
    retrieved_text = retrieved_docs.to_string(index=False)
    truncated_text = " ".join(retrieved_text.split()[:500])  # Limit to 500 words

    # ✅ Refined Prompt for Accurate Summary (won't appear in output)
    prompt = f"""
You are a medical AI assistant providing structured and professional reports based on medical records.
Use ONLY the following data to generate an informative and well-organized medical report.
DO NOT include any instructions or prompts in your response.
DO NOT make up any information not present in the data.
Respond ONLY with the structured medical report.

**User Query:** {user_query}

**Retrieved Medical Records:** {truncated_text}

Generate the report in the following format (fill each section with specific data from above):

1. **Diagnosis Summary:** [Concise diagnosis statement]
2. **Presenting Symptoms:** [Bullet points of key symptoms]
3. **Clinical Findings:** [Relevant test results and examinations]
4. **Treatment Plan:** [Current or recommended treatments]
5. **Follow-up Recommendations:** [If any mentioned in records]

Use clear, professional medical language. Keep each section concise but informative.
"""
    
    # ✅ Retry API Call if it Fails
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 600}},
                timeout=30
            )

            if response.status_code == 200:
                json_response = response.json()
                if isinstance(json_response, list):
                    # Clean up the response to remove any prompt remnants
                    generated_text = json_response[0].get("generated_text", "")
                    # Remove the prompt part if it appears in output
                    if "**User Query:**" in generated_text:
                        generated_text = generated_text.split("**User Query:**")[0]
                    return generated_text.strip()
                else:
                    return "Error: Unexpected API response format"
            elif response.status_code == 422:
                return "Error: Input too long. Please try a shorter query."
            else:
                return f"Error {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Wait before retrying
            else:
                return f"Error: API request failed after multiple attempts. {str(e)}"

# ✅ Streamlit UI
st.title("🩺 Medical AI Assistant")
st.markdown("""
<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;">
    <h4 style="color:#2e59a9;margin-top:0;">Enter a medical case or symptoms to generate a structured medical report</h4>
    <p style="color:#6c757d;">Examples: "Diabetic patient with foot pain and numbness", "45-year-old male with chest pain and shortness of breath"</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area(
        "**🔍 Enter Medical Query:**",
        placeholder="E.g., 62-year-old female with history of hypertension presenting with dizziness and blurred vision",
        help="Be as specific as possible for better results"
    )

with col2:
    st.markdown("""
    <div style="background-color:#e8f4fd;padding:15px;border-radius:10px;margin-top:10px;">
        <h5 style="color:#2e59a9;margin-top:0;">Tips for Best Results:</h5>
        <ul style="font-size:14px;padding-left:20px;">
            <li>Include patient demographics</li>
            <li>List key symptoms</li>
            <li>Mention relevant history</li>
            <li>Keep queries 10-30 words</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if st.button("Generate Report", type="primary", use_container_width=True):
    if query.strip():
        with st.spinner("🔍 Searching medical knowledge base..."):
            retrieved_results = asyncio.run(retrieve_documents(query))

        if not retrieved_results.empty:
            with st.spinner("📝 Generating comprehensive report..."):
                summary = asyncio.run(generate_medical_summary(query, retrieved_results))

            st.subheader("📄 Medical Report")
            
            # Enhanced output display
            if not summary.startswith("Error"):
                st.markdown(f"""
                <div class="report">
                    {summary.replace('**', '<span class="highlight">').replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Add download button
                st.download_button(
                    label="Download Report",
                    data=summary,
                    file_name="medical_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.error(summary)
        else:
            st.warning("⚠️ No relevant medical records found. Please try:")
            st.markdown("""
            <ul>
                <li>Using different keywords</li>
                <li>Making your query more specific</li>
                <li>Checking for spelling errors</li>
            </ul>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ Please enter a medical query to generate a report.")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6c757d;font-size:14px;">
    <p>Medical AI Assistant v1.0 | For clinical decision support only | Not a substitute for professional medical judgment</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    st.write("")  # Empty space for cleaner layout
