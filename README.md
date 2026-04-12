# 🏥 Medical Report Analyzer

An AI-powered web application that analyzes medical reports and answers questions in plain English using RAG (Retrieval Augmented Generation).

## 🚀 Demo
Upload any medical PDF → Ask questions → Get instant AI-powered answers!

## 🛠️ Tech Stack
- **Frontend:** React.js
- **Backend:** FastAPI (Python)
- **AI Model:** Groq LLaMA 3.3 70B
- **RAG Pipeline:** TF-IDF + Cosine Similarity
- **PDF Processing:** PyMuPDF

## ⚙️ How It Works
1. User uploads a medical PDF report
2. Backend extracts text using PyMuPDF
3. Text is split into chunks
4. TF-IDF finds most relevant chunks (RAG)
5. Groq LLaMA generates a simple answer

## 🏃 How To Run

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

## 📁 Project Structure
```
medical-report-analyzer/
├── backend/          # FastAPI backend
│   └── main.py      # API endpoints + RAG pipeline
├── frontend/         # React frontend
│   └── src/
│       └── App.js   # Main UI component
├── app.py           # Streamlit prototype
└── requirements.txt
```