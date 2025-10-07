# db_demo_local.py


import os
import sqlite3
import pandas as pd
import faiss
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIG
DB_PATH = "ecommerce.db"
INDEX_PATH = "vector.index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# INITIALIZE MODELS
print("Loading models...")

# Sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Local LLM model (choose one)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print(" Models loaded successfully.")

# FASTAPI APP
app = FastAPI(title="Ecommerce RAG API (Local)")

# UTILITIES
class QueryRequest(BaseModel):
    question: str

def get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Database not found! Please upload a CSV first.")
    return sqlite3.connect(DB_PATH)

def build_faiss_index():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM ecommerce", conn)
    conn.close()

    docs = df.astype(str).agg(" | ".join, axis=1).tolist()
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    return docs, index

def load_index():
    if not os.path.exists(INDEX_PATH):
        return build_faiss_index()
    index = faiss.read_index(INDEX_PATH)
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM ecommerce", conn)
    conn.close()
    docs = df.astype(str).agg(" | ".join, axis=1).tolist()
    return docs, index

def generate_local_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_answer(question: str):
    docs, index = load_index()
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k=3)
    retrieved = [docs[i] for i in I[0] if i < len(docs)]
    context = "\n".join(retrieved)

    prompt = f"""
You are a helpful assistant that answers ecommerce-related questions using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    return generate_local_response(prompt)

# ROUTES
@app.get("/")
def root():
    return {"status": "ok", "msg": "Local Ecommerce RAG API is running!"}

@app.post("/load_csv")
async def load_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("ecommerce", conn, if_exists="replace", index=False)
        conn.close()
        build_faiss_index()
        return {"status": "success", "rows_loaded": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

@app.post("/query_sql")
async def query_db(req: QueryRequest):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(req.question)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL Error: {str(e)}")

@app.post("/query_nl")
async def query_nl(req: QueryRequest):
    try:
        answer = rag_answer(req.question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"RAG Error: {str(e)}")

