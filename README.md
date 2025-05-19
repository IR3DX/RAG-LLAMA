# 🧠 PDF Question Answering with RAG AI

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using:

- 🦙 IBM Watsonx + LLaMA 3 as the language model
- 🧠 Sentence-transformers for embeddings
- 📄 Chroma for vector search
- 📚 PyPDFLoader for document parsing
- 🧩 LangChain for orchestration

The system reads a PDF, chunks the text, creates a vector store with embeddings, and enables intelligent question answering over that content.

---

## 📂 File Overview

### ✅ `worker.py`
This is the **only file authored in this repository**. It:

- Initializes the LLM and embeddings
- Loads and splits a PDF
- Builds a retrieval-based QA chain
- Accepts user questions and returns answers

### 🧑‍💻 Full Stack Bot Interface
If you're looking for the **full chatbot application** (UI, server, deployment, etc.), please check the included zipped archive
It contains everything you need to run the complete bot interface on top of this worker engine.
Credits for the fullstack: https://github.com/sinanazeri/build_own_chatbot_without_open_ai

---

## 🛠 Requirements
See requirements.txt in this repo.

---

## 🧾 License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for more details.

--- 
🤝 Acknowledgments

Special thanks for IBM's courses that taught me all these fascinating technologies and how to implement them in meaningful projects

Thanks to the developers of LangChain, IBM Watsonx, HuggingFace, ChromaDB, and PyPDFLoader for their open tools and models. 

Fullstack: https://github.com/sinanazeri/build_own_chatbot_without_open_ai
