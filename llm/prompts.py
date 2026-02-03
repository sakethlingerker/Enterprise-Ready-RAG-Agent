def qa_prompt(context, question):
    return f"""
You are an AI assistant helping analyze research papers.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
If metrics are present, quote them exactly.
"""
