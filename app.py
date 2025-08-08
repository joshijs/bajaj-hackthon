import os
import shutil
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chromadb import Client
from chromadb.config import Settings
import google.generativeai as genai

# ------------ CONFIG ------------
PDF_FOLDER = "PDFS"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "pdf_collection"
GOOGLE_API_KEY ="AIzaSyC0CXoXPpCYmFZGO_p4iw6Vo5cRb29ituQ"
genai.configure(api_key="AIzaSyC0CXoXPpCYmFZGO_p4iw6Vo5cRb29ituQ")

# ------------ INIT FASTAPI ------------
app = FastAPI()

# ------------ INIT CHROMA DB ------------
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

chroma_client = Client(Settings(
    persist_directory=DB_FOLDER,
    chroma_db_impl="duckdb+parquet"
))

collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# ------------ PDF TEXT EXTRACTION (with OCR fallback) ------------
def extract_text_with_ocr(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page_index, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text += page_text + "\n"
        else:
            images = convert_from_path(
                file_path,
                first_page=page_index + 1,
                last_page=page_index + 1
            )
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
    return text

# ------------ EMBEDDING WITH GEMINI ------------
def embed_text(text: str):
    model = "models/text-embedding-004"
    embedding = genai.embed_content(model=model, content=text)["embedding"]
    return embedding

# ------------ LOAD PDFs INTO CHROMA ------------
def load_pdfs_into_chroma():
    collection.delete(where={})  # Clear old data
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            text = extract_text_with_ocr(file_path)
            if text.strip():
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                for idx, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk],
                        embeddings=[embed_text(chunk)],
                        ids=[f"{filename}_{idx}"]
                    )

# ------------ API MODELS ------------
class QueryRequest(BaseModel):
    question: str

# ------------ API ENDPOINTS ------------
@app.on_event("startup")
def startup_event():
    load_pdfs_into_chroma()

@app.post("/query")
def query_rag(req: QueryRequest):
    results = collection.query(
        query_embeddings=[embed_text(req.question)],
        n_results=3
    )
    if not results["documents"]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context = "\n".join(doc for sublist in results["documents"] for doc in sublist)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Answer the question using the following context:\n{context}\nQuestion: {req.question}")
    
    return JSONResponse({"answer": response.text})

@app.post("/reload")
def reload_data():
    load_pdfs_into_chroma()
    return {"status": "PDF data reloaded"}

# ------------ CLEANUP (optional) ------------
@app.on_event("shutdown")
def shutdown_event():
    chroma_client.persist()
