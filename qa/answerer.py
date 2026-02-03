from qa.retriever import DocumentRetriever
from llm.llm_client import ask_llm

TOP_K = 6
MAX_CONTEXT_CHARS = 12000

def generate_answer(question: str, retriever: DocumentRetriever, chunks: list, chat_history: list = None, model_name: str = "llama-3.1-8b-instant") -> tuple[str, list]:
    """
    Generates an answer to the question based on the provided chunks, retriever, and chat history.
    
    Args:
        question (str): The user's question.
        retriever (DocumentRetriever): The initialized retriever instance.
        chunks (list): List of all document chunks.
        chat_history (list): List of dicts with 'role' and 'content'.
        model_name (str): The model to use for generation.
        
    Returns:
        tuple[str, list]: The generated answer and the list of source chunks (hits).
    """
    # Retrieve relevant chunks
    hits = retriever.retrieve(question, top_k=TOP_K)
    
    # Conclusion handling
    if "conclusion" in question.lower():
        hits = chunks[-8:] + hits
    
    # Build context
    context_parts = []
    used_sources = []
    for chunk in hits:
        content = chunk.get("content", "")
        page = chunk.get("page", "N/A")
        context_parts.append(f"[Page {page}]\n{content}")
        
        # Track sources for citation
        used_sources.append({
            "page": page,
            "content": content,
            "section": chunk.get("section", "unknown")
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Limit context size
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."
    
    # Format chat history
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]: 
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Generate answer
    prompt = f"""Answer the question based ONLY on the provided document context and chat history.

DOCUMENT CONTEXT:
{context}

CHAT HISTORY:
{history_text}

QUESTION: {question}

Answer based on the context and history above. If the information is not in the context, say so.
ANSWER:"""
    
    answer = ask_llm(prompt, model=model_name)
    return answer, used_sources
