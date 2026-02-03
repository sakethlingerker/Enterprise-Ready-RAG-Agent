import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Research-grade LLM call.
    Forces confident academic inference instead of refusal.
    """

    system_prompt = """
You are a senior research assistant reviewing academic papers.

STRICT RULES (MANDATORY):
- Answer ONLY using the provided context.
- NEVER say:
  "cannot determine", "insufficient information",
  "it appears to be", "seems like", or "not enough context".
- If a conclusion section is not explicitly present,
  INFER the conclusion from results, evaluation, and discussion.
- Clearly state when the conclusion is inferred.
- Do NOT describe tables or references unless explicitly asked.
- Be confident, technical, and concise.
- Do NOT hallucinate facts not supported by the context.

Behave like a human researcher summarizing a paper for review.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()
