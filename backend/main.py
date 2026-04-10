from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz
import os

load_dotenv()

app = FastAPI(title="Medical Report Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

@app.get("/health")
def health_check():
    return {"status": "Server chal raha hai!"}

@app.post("/analyze")
async def analyze_report(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # Step 1: PDF se text nikalo
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Step 2: Chunks banao
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Step 3: RAG — relevant chunks dhundho
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])

    # Step 4: AI se jawab lo
    prompt = f"""You are a helpful medical assistant.
Below is the relevant content from a medical report:

{context}

Question: {question}

Answer in simple, easy to understand English.
Always remind the user to consult a doctor."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"answer": response.content}