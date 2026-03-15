import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def initialize_llm(model_name: str | None = None):
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name or os.getenv(
            "GROQ_MODEL", "llama-3.1-8b-instant"
        ),
        temperature=0.2,
        timeout=60
    )
