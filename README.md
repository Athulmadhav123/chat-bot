# chat-bot
# AI Customer Support Bot using RAG, FAISS & Groq

An AI-powered customer support chatbot that answers customer queries using company documents and escalates unknown queries to a human agent.

This project uses:

- Flask
- Groq API
- LangChain
- FAISS
- HuggingFace Embeddings

---

# Features

- Train chatbot on company/client documents
- Answer L1 customer support queries
- Semantic document search using FAISS
- AI-generated responses using Groq LLM
- Human escalation for unsupported queries
- Simple Flask web interface

---

# Project Workflow

```text
User Question
      ↓
Search Company Documents
      ↓
Retrieve Relevant Context
      ↓
Generate AI Response
      ↓
Confidence Check
   ↓            ↓
Good         Low confidence
 ↓                 ↓
Answer        Escalate to Human
