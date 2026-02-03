# 🤖 Enterprise Document Q&A AI Agent

An advanced, enterprise-grade AI agent demonstrating **RAG (Retrieval Augmented Generation)**, **Multi-Modal Ingestion**, and **Agentic Capabilities** using open LLMs via Groq.

## 🚀 Key Features

### 🧠 Advanced NLP

- **Multi-Model Support**: Switch between **Llama 3.1**, **Llama 3.3 (70B)**, and others instantly.
- **Semantic Chunking**: Uses recursive character splitting to preserve paragraph/sentence meaning.
- **Context-Aware Memory**: Remembers previous chat turns for natural follow-up questions.

### 📄 Enterprise Ingestion Pipeline

- **PDF & Table Extraction**: Parses text and **restores tables** to structured Markdown for data analysis tasks.
- **Source Citations**: Every answer provides expandable verification links citing specific pages and sections.
- **Sections Support**: Detects `Abstract`, `Conclusion`, `Results` for targeted retrieval.

### 🕵️‍♂️ Agentic Capabilities

- **Arxiv Research Agent**: Type _"Find papers on Transformers"_ or _"Give me papers on RAG"_ in chat. The agent will:
  1. Detect your intent.
  2. Search the arXiv API.
  3. Propose papers.
  4. **Download & Process** them automatically upon request.

## 🛠️ Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file:

   ```env
   GROQ_API_KEY=gsk_your_groq_key_here
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment (Streamlit Cloud)

1. **Push to GitHub**: Ensure your code is on GitHub.
2. **Login**: Go to [share.streamlit.io](https://share.streamlit.io/) and login with GitHub.
3. **Deploy**:
   - Click **"New App"**.
   - Select the `sakethlingerker/Enterprise-Ready-RAG-Agent` repository.
   - Set **Main file path** to `app.py`.
4. **Secrets (Crucial Step)**:
   - Click **"Advanced Settings"** -> **"Secrets"**.
   - Add your API key:
     ```toml
     GROQ_API_KEY = "gsk_..."
     ```
   - Click **Save** and **Deploy**.

## 📂 Architecture

- **`ingestion/`**:
  - `pdf_loader.py`: PyMuPDF-based extraction (Text + Tables).
  - `chunker.py`: Recursive semantic splitting.
- **`qa/`**:
  - `retriever.py`: FAISS/ChromaDB vector store management.
  - `answerer.py`: Context assembly and citation logic.
- **`llm/`**:
  - `llm_client.py`: Robust client for Groq API (Llama models).
- **`app.py`**: Streamlit UI with Chat Loop and Agent Router.

## 📝 Usage Guide

1. **Upload**: Drag & Drop a PDF in the sidebar.
2. **Search**: Or type natural commands like _"Give me papers on transformers"_ or _"Find papers on RAG"_ in the chat.
3. **Chat**: Ask questions like _"What is the specific accuracy in Table 3?"_.
4. **Verify**: Check the **"📚 Source Citations"** dropdown to see the source text.
