from groq import Groq
import os
from dotenv import load_dotenv

def get_groq_client():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve API key from environment
    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")

    # Return initialized Groq client
    return Groq(api_key=key)
