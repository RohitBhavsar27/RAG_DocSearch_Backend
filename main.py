# main.py
import io
import os
import asyncio
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

# --- logging helps see where it stalls
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag_api")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        content = {"detail": jsonable_encoder(exc.errors())}
    except UnicodeDecodeError:
        content = {
            "detail": [
                {
                    "type": "validation_error",
                    "msg": "The request body contains invalid characters that could not be decoded.",
                }
            ]
        }
    return JSONResponse(status_code=422, content=content)


# CORS
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",  # <- add this if you sometimes use 127.0.0.1
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/api/health")
def read_health_check():
    return {"status": "ok", "message": "API is running!"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Google API key not found.")

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF."
        )

    try:
        pdf_content = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_content))

        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += (
                    page_text.encode("utf-8", "ignore").decode("utf-8") + "\n"
                )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_text(extracted_text)

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = embeddings_model.embed_documents(chunks)

        vector_store = FAISS.from_embeddings(
            text_embeddings=zip(chunks, embeddings),
            embedding=embeddings_model,
        )
        vector_store.save_local("faiss_index")

        log.info("Saved FAISS with %d chunks", len(chunks))
        return {
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "Vector store created and saved successfully.",
        }
    except Exception as e:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# ------------------ CHAT (async) ------------------
CHAT_TIMEOUT_S = 45


@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Answer a question using the FAISS index + Gemini."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Google API key not found.")

    try:
        log.info("Chat question: %s", request.question)

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # load index (allow_dangerous_deserialization = required by FAISS pickle)
        vector_store = FAISS.load_local(
            "faiss_index", embeddings_model, allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt_template = """
        Answer the question as detailed as possible based on the provided context.
        If the answer is not in the provided context, say "The answer is not available in the context".

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

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        # ✅ use the async variant & add a timeout
        answer = await asyncio.wait_for(
            rag_chain.ainvoke(request.question), timeout=CHAT_TIMEOUT_S
        )
        log.info(
            "Chat answer length: %d chars",
            len(answer) if isinstance(answer, str) else -1,
        )
        return {"answer": answer}

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="FAISS index not found. Please upload a document first.",
        )
    except asyncio.TimeoutError:
        log.error("Chat timed out after %ss", CHAT_TIMEOUT_S)
        raise HTTPException(
            status_code=504,
            detail="The model took too long to respond. Please try again.",
        )
    except Exception as e:
        log.exception("Chat failed")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during chat: {str(e)}"
        )
