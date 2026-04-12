# Medical Report Analyzer

An AI-powered web application that analyzes medical reports and answers questions in plain English using RAG (Retrieval Augmented Generation).

## Overview
Upload any medical PDF report, ask a question, and get an instant AI-powered answer in simple language.

## Tech Stack
- Frontend: React.js
- Backend: FastAPI (Python)
- AI Model: Groq LLaMA 3.3 70B
- RAG Pipeline: TF-IDF with Cosine Similarity
- PDF Processing: PyMuPDF

## How It Works
1. User uploads a medical PDF report
2. Backend extracts text using PyMuPDF
3. Text is split into chunks using LangChain
4. TF-IDF retrieves the most relevant chunks
5. Groq LLaMA generates a simple English answer

## How To Run

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Project Structure
```
medical-report-analyzer/
├── backend/
│   └── main.py
├── frontend/
│   └── src/
│       └── App.js
├── app.py
└── requirements.txt
```