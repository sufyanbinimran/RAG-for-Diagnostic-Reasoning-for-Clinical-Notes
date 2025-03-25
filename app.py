# ✅ Import Necessary Libraries
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# ✅ Load Medical LLM (BioGPT-Large)
model_name = "microsoft/BioGPT-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = AutoModelForCausalLM.from_pretrained(model_name)

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

# ✅ Generate Missing Fields Using BioGPT-Large
def generate_missing_info(field_name, user_query, retrieved_text):
    prompt = f"""
    You are a medical AI assistant. Given a user query and retrieved medical records, generate the missing field: **{field_name}**.

    **User Query:** {user_query}
    **Retrieved Medical Records:** {retrieved_text}

    **{field_name}:** (Provide a well-structured response)
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = generator.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Extract & Complete Structured Report
def generate_structured_report(user_query, retrieved_docs):
    report = {}
    
    # ✅ Get the actual column names from the retrieved DataFrame
    existing_columns = retrieved_docs.columns.tolist()

    # ✅ Define field-to-column mapping
    field_mapping = {
        "Diagnosis": "diagnosis",
        "Symptoms": "symptoms",
        "Medical Details": "medical_details",
        "Treatment & Cure": "treatment",
        "Physical Examination Findings": "physical_exam",
        "Patient Information": "patient_info",
        "History": "history",
        "Physical Examination": "physical_exam",
        "Diagnostic Tests": "diagnostic_tests",
        "Treatment & Management": "treatment_management",
        "Follow-Up Care": "follow_up",
        "Outlook": "outlook"
    }

    for field, column_name in field_mapping.items():
        if column_name in existing_columns:
            retrieved_text = retrieved_docs[column_name].astype(str).to_string(index=False).strip()
            
            # ✅ Use LLM only if data is missing
            if not retrieved_text or retrieved_text.lower() in ["n/a", "unknown", ""]:
                report[field] = generate_missing_info(field, user_query, retrieved_docs.to_string(index=False))
            else:
                report[field] = retrieved_text
        else:
            # ✅ Column not found → Use LLM
            report[field] = generate_missing_info(field, user_query, retrieved_docs.to_string(index=False))

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
