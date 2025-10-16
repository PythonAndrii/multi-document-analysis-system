from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
Use the following context to answer the question. If you don't know the answer, say "I don't know".

Context:
{context}

Question:
{question}
"""

def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_PROMPT)
