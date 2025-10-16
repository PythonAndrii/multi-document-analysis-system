from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Initializes and returns the Gemini LLM."""
    # Let the user to setup the model
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
