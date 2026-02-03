import streamlit as st
import tempfile
import os

from ingestion.pdf_loader import extract_pdf_content
from ingestion.chunker import chunk_text
from qa.retriever import DocumentRetriever
from qa.answerer import generate_answer
from llm.llm_client import ask_llm
from arxiv.arxiv_client import search_arxiv, download_pdf


# ===================== CONFIG =====================
TOP_K = 6


# ===================== INIT SESSION =====================
def init_session():
    """Initialize minimal session state"""
    if "active_chunks" not in st.session_state:
        st.session_state.active_chunks = None
    if "active_doc_name" not in st.session_state:
        st.session_state.active_doc_name = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "arxiv_results" not in st.session_state:
        st.session_state.arxiv_results = []


# ===================== CORE FUNCTIONS =====================
@st.cache_data(show_spinner=False)
def process_pdf_bytes(pdf_bytes: bytes, _filename: str):
    """Process PDF from bytes - cached for efficiency"""
    # Now using the updated extract_pdf_content which handles bytes directly
    sections = extract_pdf_content(pdf_bytes)
    return chunk_text(sections)


def set_active_document(chunks, name):
    """Set a document as active"""
    st.session_state.active_chunks = chunks
    st.session_state.active_doc_name = name
    st.session_state.retriever = None  # Reset retriever for new doc


def clear_all():
    """Clear everything"""
    st.cache_data.clear()
    st.session_state.active_chunks = None
    st.session_state.active_doc_name = None
    st.session_state.retriever = None
    st.session_state.arxiv_results = []
    st.session_state.messages = []
    st.rerun()


# ===================== ARXIV DOWNLOAD =====================
def download_and_process_arxiv(paper):
    """Download arXiv paper and process it"""
    with st.spinner("Downloading paper..."):
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                temp_path = tmp.name
            
            # Download
            success = download_pdf(paper["pdf_url"], temp_path)
            
            if success and os.path.exists(temp_path):
                # Read and process using path (extract_pdf_content handles paths too)
                sections = extract_pdf_content(temp_path)
                chunks = chunk_text(sections)
                set_active_document(chunks, paper["title"])
                
                return True
            else:
                st.error("Failed to download paper")
                return False
                
        except Exception as e:
            st.error(f"Download error: {str(e)}")
            return False
        finally:
            # Clean up temp file
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass


# ===================== ANSWER GENERATION =====================
def get_answer(question: str, model_name: str) -> tuple[str, list]:
    """Generate answer using current document"""
    if not st.session_state.retriever:
        # Build retriever if needed
        retriever = DocumentRetriever()
        retriever.build_index(st.session_state.active_chunks)
        st.session_state.retriever = retriever
    
    # Use the logic moved to qa.answerer
    return generate_answer(
        question=question,
        retriever=st.session_state.retriever,
        chunks=st.session_state.active_chunks,
        chat_history=st.session_state.messages[-6:-1] if "messages" in st.session_state else None,
        model_name=model_name
    )


# ===================== MAIN UI =====================
def main():
    # Setup
    init_session()
    st.set_page_config("Document Q&A", "📄", layout="wide")
    
    # Title
    st.title("📄 Document Q&A")
    st.markdown("---")
    
    # Sidebar for Model Selection
    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.selectbox(
            "Select Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            index=0
        )
        
        # Model descriptions
        if model_name == "llama-3.1-8b-instant":
            st.caption("🚀 **Fast & Efficient**: Best for quick summaries and simple queries.")
        elif model_name == "llama-3.3-70b-versatile":
            st.caption("🧠 **Smart & Detailed**: Best for complex reasoning and deep research.")
            
        st.divider()
        st.markdown("**Note:** Ensure you have enough rate limits for larger models.")

    # Two-column layout
    col_left, col_right = st.columns([1.3, 1])
    
    # ========== LEFT COLUMN: Document Management ==========
    with col_left:
        # Upload section
        st.subheader("📤 Upload PDF")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        
        if uploaded_pdf:
            chunks = process_pdf_bytes(uploaded_pdf.getvalue(), uploaded_pdf.name)
            set_active_document(chunks, uploaded_pdf.name)
            st.success(f"✅ '{uploaded_pdf.name}' loaded")
        
        st.divider()
        
        # arXiv Search section
        st.subheader("🔍 Search arXiv")
        
        arxiv_query = st.text_input(
            "Search for papers",
            placeholder="e.g., machine learning",
            label_visibility="collapsed"
        )
        
        if st.button("Search arXiv", use_container_width=True) and arxiv_query:
            with st.spinner("Searching..."):
                results = search_arxiv(arxiv_query, max_results=5)
                st.session_state.arxiv_results = results if results else []
        
        # Display arXiv results
        if st.session_state.arxiv_results:
            st.write(f"**Found {len(st.session_state.arxiv_results)} papers**")
            
            for i, paper in enumerate(st.session_state.arxiv_results):
                with st.expander(f"📄 {paper.get('title', '')[:80]}..."):
                    # Paper info
                    authors = paper.get('authors', [])
                    if authors:
                        st.caption(f"**Authors:** {', '.join(authors[:3])}")
                    
                    # Summary
                    summary = paper.get('summary', '')
                    if len(summary) > 300:
                        st.write(summary[:300] + "...")
                    else:
                        st.write(summary)
                    
                    # Use button
                    if st.button("📥 Use this paper", key=f"arxiv_{i}"):
                        if download_and_process_arxiv(paper):
                            st.success("Paper loaded successfully!")
                            st.rerun()
        
        st.divider()
        
        # Clear button
        if st.button("🧹 Clear All", use_container_width=True, type="secondary"):
            clear_all()
    
    # ========== RIGHT COLUMN: Chat Interface ==========
    with col_right:
        st.subheader("💬 Chat with Document")
        
        # Document status
        if not st.session_state.active_chunks:
            st.info("👈 Upload a PDF or search arXiv first")
        else:
            # Show active document
            st.success(f"📄 **Active:** {st.session_state.active_doc_name}")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("Ask a question about the document..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # 🕵️‍♂️ Agentic Intent Detection (Simple Router)
                            lower_prompt = prompt.lower()
                            search_triggers = ["find paper", "search arxiv", "search for paper", "look up paper", "get paper", "give me paper", "show paper"]
                            if any(k in lower_prompt for k in search_triggers):
                                # Extract query (naive removal of keywords)
                                query = lower_prompt
                                for term in search_triggers:
                                    query = query.replace(term, "")
                                query = query.replace("s ", " ").strip() # handle plurals naturally
                                
                                st.markdown(f"🔎 **Searching arXiv for:** *{query}*...")
                                
                                # Call Tool
                                results = search_arxiv(query, max_results=3)
                                
                                if not results:
                                    answer = f"I couldn't find any papers matching '{query}'."
                                    st.markdown(answer)
                                else:
                                    answer = f"I found {len(results)} papers for '{query}':"
                                    st.markdown(answer)
                                    
                                    for i, paper in enumerate(results):
                                        with st.expander(f"📄 {paper['title']}"):
                                            st.write(paper['summary'][:200] + "...")
                                            if st.button(f"📥 Download '{paper['title'][:20]}...'", key=f"chat_dl_{i}"):
                                                if download_and_process_arxiv(paper):
                                                    st.success("Downloaded & Processed! You can now ask questions about it.")
                                                    st.rerun()
                            
                            else:
                                # Standard RAG Response
                                answer, citations = get_answer(prompt, model_name)
                                st.markdown(answer)
                                
                                # Display Citations
                                if citations:
                                    with st.expander("📚 Source Citations"):
                                        for i, cit in enumerate(citations):
                                            st.markdown(f"**Source {i+1}** (Page {cit['page']}, Section: {cit['section']})")
                                            st.caption(cit['content'][:200] + "...")
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

