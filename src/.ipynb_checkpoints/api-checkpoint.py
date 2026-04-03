from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_lcel import build_chain
import os
from pathlib import Path
from dotenv import load_dotenv

# ✅ LOAD ENV VARIABLES
#env_path = Path(__file__).resolve().parent / ".env"
#load_dotenv(dotenv_path=env_path)
load_dotenv()
app = FastAPI()

chain = build_chain(where=None)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    result = chain.invoke(req.question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "grounding": result["grounding"]
    }

@app.get("/")
def health():
    return {"status": "RAG API running"}