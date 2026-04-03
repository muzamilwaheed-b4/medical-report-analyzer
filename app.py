import streamlit as st
import fitz
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # ← Yeh SABSE PEHLE hona chahiye!

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

st.set_page_config(page_title="Medical Report Analyzer", page_icon="🏥")
st.title("🏥 Medical Report Analyzer")
st.write("Upload your medical report — I'll explain it in simple language!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading your report..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
    st.success("✅ Report loaded successfully!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)

    def ask_question(question):
        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, chunk_vectors)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])
        prompt = f"""You are a helpful medical assistant.
Below is the relevant content from a medical report:

{context}

Question: {question}

Answer in simple, easy to understand English.
Always remind the user to consult a doctor."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    st.divider()
    st.subheader("💬 Ask a Question")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Overall Report Summary"):
            with st.spinner("Thinking..."):
                st.info(ask_question("Give a complete summary of this medical report. Mention any concerning values."))
    with col2:
        if st.button("⚠️ Any Abnormal Values?"):
            with st.spinner("Thinking..."):
                st.warning(ask_question("List all values that are outside the normal range and explain what they mean."))

    st.divider()
    question = st.text_input("Or type your own question:")
    if question:
        with st.spinner("Finding answer..."):
            st.success(ask_question(question))