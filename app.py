import os
import glob
import tempfile
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import google.generativeai as genai
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# ===== CONFIG =====
PDF_FOLDER = "PDFS"
GOOGLE_API_KEY ="AIzaSyC0CXoXPpCYmFZGO_p4iw6Vo5cRb29ituQ"

# Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ===== PDF LOADING WITH OCR =====
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from both normal PDF text and images inside PDF using OCR."""
    reader = PdfReader(file_path)
    text = ""

    # 1️⃣ Extract selectable text
    for page in reader.pages:
        text += page.extract_text() or ""

    # 2️⃣ OCR for images in PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(file_path, output_folder=temp_dir)
        for img in images:
            text += "\n" + pytesseract.image_to_string(img)

    return text.strip()

def load_pdfs(folder_path: str) -> List[Document]:
    docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        text = extract_text_from_pdf(file_path)
        if text:
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))
    return docs

# ===== CHROMA IN-MEMORY =====
def build_chroma_in_memory() -> Chroma:
    docs = load_pdfs(PDF_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=None  # ✅ In-memory
    )
    return vectorstore

vectorstore = build_chroma_in_memory()

# ===== FASTAPI APP =====
app = FastAPI(title="PDF RAG with OCR + Gemini (In-Memory)")

class QueryRequest(BaseModel):
    question: str
    k: int = 3

@app.get("/")
def home():
    return {"message": "PDF RAG API with OCR is running!"}

@app.post("/ask")
def ask_question(req: QueryRequest):
    docs = vectorstore.similarity_search(req.question, k=req.k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. Use the provided context to answer the question.
    Context:
    {context}

    Question: {req.question}
    Answer:
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return {
        "question": req.question,
        "answer": response.text,
        "sources": [d.metadata["source"] for d in docs]
    }
