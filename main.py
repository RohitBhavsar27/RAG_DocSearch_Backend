# main.py
import io
import os
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pypdf
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# ——————————————————————————————————————————
# Paths
PDF_DIR = "storage/pdfs"
INDEX_DIR = "storage/indices"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ——————————————————————————————————————————
app = FastAPI()

# CORS (Angular dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve PDFs for the viewer
app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")


# Validation handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        content = {"detail": jsonable_encoder(exc.errors())}
    except UnicodeDecodeError:
        content = {
            "detail": [
                {"type": "validation_error", "msg": "Invalid characters in body."}
            ]
        }
    return JSONResponse(status_code=422, content=content)


# Models
class ChatRequest(BaseModel):
    question: str
    doc_id: str


# Health
@app.get("/api/health")
def read_health_check():
    return {"status": "ok", "message": "API is running!"}


# ——————————————————————————————————————————
# Helpers
def index_path(doc_id: str) -> str:
    return os.path.join(INDEX_DIR, doc_id)


def pdf_path(doc_id: str) -> str:
    return os.path.join(PDF_DIR, f"{doc_id}.pdf")


# ——————————————————————————————————————————
# Upload: save PDF, build FAISS with page metadata, return doc_id
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Google API key not found.")
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF."
        )

    try:
        # Generate an id and save the raw PDF for preview
        doc_id = uuid.uuid4().hex
        pdf_file_path = pdf_path(doc_id)
        raw = await file.read()
        with open(pdf_file_path, "wb") as f:
            f.write(raw)

        # Extract text per page
        reader = pypdf.PdfReader(io.BytesIO(raw))
        per_page_text: List[str] = []
        for p in reader.pages:
            t = (p.extract_text() or "").encode("utf-8", "ignore").decode("utf-8")
            per_page_text.append(t)

        # Split per page (so we keep page numbers in metadata)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs: List[Document] = []
        for idx, page_text in enumerate(per_page_text, start=1):
            if not page_text.strip():
                continue
            chunks = splitter.split_text(page_text)
            for ch in chunks:
                docs.append(
                    Document(
                        page_content=ch,
                        metadata={
                            "page": idx,
                            "doc_id": doc_id,
                            "filename": file.filename,
                        },
                    )
                )

        # Build FAISS index for this document
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(index_path(doc_id))

        return {
            "filename": file.filename,
            "chunks_created": len(docs),
            "status": "Vector store created and saved successfully.",
            "doc_id": doc_id,
            "pdf_url": f"/pdfs/{doc_id}.pdf",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# ——————————————————————————————————————————
# Chat: load index for doc_id, answer with citations
@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vs = FAISS.load_local(
            index_path(request.doc_id),
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
        retriever = vs.as_retriever(search_kwargs={"k": 3})

        prompt_template = """
        Answer the question as detailed as possible based on the provided context.
        If the answer is not in the provided context, say: "The answer is not available in the context".
        Don't invent facts.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(d.page_content for d in docs)

        # ✅ retriever is a Runnable; wrap the formatter with RunnableLambda
        rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | StrOutputParser()
        )

        # Get docs for citations and run the chain for the answer
        retrieved_docs = retriever.get_relevant_documents(request.question)
        answer = rag_chain.invoke(request.question)

        citations = []
        for d in retrieved_docs:
            md = d.metadata or {}
            citations.append(
                {
                    "title": md.get("filename") or "Document",
                    "page": int(md.get("page") or 1),
                    "snippet": (d.page_content or "")[:400],
                    "pdf_url": f"/pdfs/{md.get('doc_id')}.pdf",
                }
            )

        return {"answer": answer, "citations": citations}

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="FAISS index not found. Please upload a document first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during chat: {str(e)}"
        )
