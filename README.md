# DocuMentor — RAG Chatbot API (FastAPI + FAISS)
FastAPI backend for a RAG chatbot that lets users upload PDFs, ask questions, and receive grounded answers with page-level citations. On upload, we build a per-document FAISS index; PDFs are served via `/pdfs/{doc_id}.pdf`. The chat endpoint returns `{ answer, citations[] }`.

This repo contains the backend only. The Angular UI lives in a separate repository.

## ✨ Features
- Upload PDFs `(/api/upload)`: extract text per page, split into chunks, generate embeddings, and index with FAISS.
- Serve PDFs `(/pdfs/*)`: static route to preview PDFs inline in the frontend.
- Chat with context `(/api/chat)`: retrieve relevant chunks → prompt LLM → return answer + citations (page/snippet/pdf_url).
- Health check `(/api/health)`.
- Robust input handling: custom validation error responses; UTF-8 text cleaning.

## 🧱 Tech Stack
- FastAPI, Uvicorn/Gunicorn
- LangChain (LCEL) for chaining; RunnableLambda fix applied
- Google Generative AI Embeddings `models/embedding-001`
- Gemini 1.5 Flash for answers
- FAISS (per-doc index)
- PyPDF for text extraction
- CORS middleware for local/dev cross-origin use

## ⚙️ Requirements
- Python 3.11+
- A Google API key with access to Generative AI models

`requirements.txt` 
```
fastapi==0.115.0
uvicorn==0.30.6
gunicorn==22.0.0
python-dotenv==1.0.1
pypdf==4.3.1
python-multipart==0.0.9
faiss-cpu==1.8.0.post1
langchain==0.2.12
langchain-community==0.2.11
langchain-google-genai==1.0.8
```

## 🔐 Environment
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

## 🚀 Run locally
```
pip install -r requirements.txt

# dev server (auto-reload)
uvicorn main:app --reload

# or production-ish
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 main:app
```
By default:
- API base: `http://127.0.0.1:8000/api`
- PDFs served at: `http://127.0.0.1:8000/pdfs/<doc_id>.pdf`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## 🧩 Endpoints
`GET /api/health`

Simple liveness probe.
```
{ "status": "ok", "message": "API is running!" }
```

`POST /api/upload`

Accepts a single `application/pdf` file via multipart form.
- Validates content type.
- Extracts text per page (PyPDF) → splits (RecursiveCharacterTextSplitter).
- Embeds with `models/embedding-001`.
- Builds a FAISS index for this doc and saves it locally (per doc).
- Saves original PDF (for inline preview) and returns `doc_id`.

Response
```
{
  "filename": "report.pdf",
  "chunks_created": 33,
  "status": "Vector store created and saved successfully.",
  "doc_id": "abc123hex",
  "pdf_url": "/pdfs/abc123hex.pdf"
}
```

`POST /api/chat`

Body:
```
{
  "question": "What is the executive summary?",
  "doc_id": "abc123hex"
}
```

Process:

- Loads FAISS for `doc_id`, retrieves top-k chunks.
- LCEL prompt (PromptTemplate → ChatGoogleGenerativeAI → StrOutputParser).
- Returns answer + citations built from doc metadata.

Response
```
{
  "answer": "…",
  "citations": [
    { "title": "report.pdf", "page": 2, "snippet": "…", "pdf_url": "/pdfs/abc123hex.pdf" }
  ]
}
```

## 🧠 Key Implementation Notes

- LCEL fix: pipe retriever into a formatter with `RunnableLambda`:
```
rag_chain = (
  { "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough() }
  | prompt
  | model
  | StrOutputParser()
)
```

- Per-doc FAISS: one index per `doc_id` ensures clean isolation.
- Page metadata: we keep `page`, `filename`, `doc_id` in metadata for accurate citations.
- Static files: PDFs are served via `app.mount("/pdfs", StaticFiles(directory=PDF_DIR))`.

## 🔌 CORS
For local dev (Angular on http://localhost:4200), enable:
```
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:4200"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
```
