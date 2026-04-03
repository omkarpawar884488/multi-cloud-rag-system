# 🌐 Multi-Cloud RAG System (AWS • Azure • GCP)

A production-style **Retrieval-Augmented Generation (RAG)** system that ingests, processes, and queries cloud architecture documentation across **AWS, Azure, and GCP**, with advanced evaluation and semantic grounding.

---

## 🚀 Overview

This project implements an **end-to-end multi-cloud RAG pipeline** that allows users to ask natural language questions about AWS, Azure, and GCP and receive accurate, source-backed answers based on official cloud documentation and best practices.

The system is designed for cloud engineers, architects, and data professionals to quickly retrieve reliable cloud architecture knowledge across multiple providers.

## 🎯 Scope of the Project

This system focuses on answering questions related to:

• Architecture & design patterns  
• Best practices (Well-Architected frameworks)  
• Service fundamentals (EC2, VNet, IAM, etc.)  
• Security & compliance  
• Cost optimization & governance  
• Cross-cloud comparisons (AWS vs Azure vs GCP)  
• Real-world architecture examples and reference architectures  
• Cloud migration and modernization strategies  


Unlike basic RAG implementations, this system includes:

- 🔍 Multi-source ingestion (Web + GitHub)
- 🧠 Topic-aware retrieval & deterministic reranking
- 📊 Evaluation with heuristics + LLM-as-a-judge
- ⚙️ Modular, reproducible pipeline
- 🌐 API + UI interface


---

## 🏗️ Architecture

```
User Query
↓
Retriever (Vector DB - Chroma)
↓
Topic-Aware Scoring (Keyword-based)
↓
Deterministic Reranking
↓
Top-K Context Selection
↓
LLM (OpenAI)
↓
Answer Generation
↓
Evaluation (Grounding + Coverage + LLM Judge)
```

---

## ⚙️ Features

### 🔹 Multi-Cloud Data Ingestion
- AWS, Azure, GCP documentation crawling
- GitHub markdown extraction
- Domain-aware crawling with filtering

### 🔹 Data Processing Pipeline
- Deduplication & normalization
- Intelligent chunking (sliding window)
- Metadata enrichment (provider, category, source)

### 🔹 Vector Database
- ChromaDB (persistent storage)
- SentenceTransformers embeddings
- Efficient similarity search

### 🔹 Advanced Retrieval
- Topic-aware filtering
- Provider-specific keyword scoring
- Deterministic reranking (custom implementation)

### 🔹 RAG Pipeline
- LangChain Expression Language (LCEL)
- Context building with structured prompts
- Source-grounded responses

### 🔹 Evaluation Framework
- Heuristic metrics:
  - Topic coverage
  - Provider coverage
  - Semantic grounding
- LLM-as-a-judge:
  - Correctness
  - Relevance
  - Completeness

### 🔹 Interfaces
- ⚡ FastAPI backend
- 🌐 Streamlit UI

---

## 📁 Project Structure

```
Rag_cloud_platforms/
│
├── src/                # Core pipeline & RAG logic
├── config/             # Source configurations
├── notebooks/
│   ├── rag_demo.ipynb
│   ├── rag_eval_final.ipynb
├── app.py              # Streamlit UI
├── README.md
```


---

## ⚠️ Note on Data

Due to large dataset size (~9GB):

- Raw data and vector database are NOT included
- The system is fully reproducible via pipeline scripts

---

## ▶️ How to Run

### 1️⃣ Install dependencies

Install required libraries manually:

```bash
pip install streamlit fastapi uvicorn requests chromadb sentence-transformers langchain openai beautifulsoup4
```

### 2️⃣ Set environment variables
Create `.env` file:

```
OPENAI_API_KEY=your_api_key
API_URL=http://127.0.0.1:8000
```

### 3️⃣ Start API
```bash
uvicorn src.api:app --reload
```

### 4️⃣ Run UI
```bash
streamlit run app.py
```

---

## 💬 Example Query

"What is AWS Well-Architected Framework and how does it compare with Azure Landing Zones?"


## 🧠 Key Insights
Topic-aware retrieval improves relevance across providers
Deterministic reranking ensures explainability
LLM-based evaluation helps detect hallucinations
Multi-cloud context improves answer completeness

---

## 🛠️ Tech Stack
- Python
- LangChain (LCEL)
- ChromaDB
- SentenceTransformers
- OpenAI API
- FastAPI
- Streamlit
- BeautifulSoup

---

## 📌 Future Improvements
Cross-encoder reranking (ML-based)
Deployment on AWS (EC2 + S3)
Streaming responses
UI enhancements

---

👨‍💻 Author

Omkar Pawar
Data Scientist | GenAI & RAG Systems

⭐ If you like this project

Give it a ⭐ on GitHub!
