import os
import glob
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# ===== CONFIG =====
PDF_FOLDER = "PDFS"
CHROMA_DIR = "/data/chroma_db"  # Persistent disk mount on Render
GOOGLE_API_KEY ="AIzaSyC0CXoXPpCYmFZGO_p4iw6Vo5cRb29ituQ"

# Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY   # ✅ force API key instead of ADC
)
# ===== PDF LOADING =====
def load_pdfs(folder_path: str) -> List[Document]:
    docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))
    return docs

# ===== CHROMA SETUP =====
def build_or_load_chroma() -> Chroma:
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("✅ Loading existing ChromaDB")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    else:
        print("📄 No DB found, indexing PDFs...")
        docs = load_pdfs(PDF_FOLDER)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
        print("✅ ChromaDB built and persisted")
        return vectorstore

vectorstore = build_or_load_chroma()

# ===== FASTAPI APP =====
app = FastAPI(title="PDF RAG with Gemini + Chroma (Render Compatible)")

class QueryRequest(BaseModel):
    question: str
    k: int = 3

@app.get("/")
def home():
    return {"message": "PDF RAG API is running!"}

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


