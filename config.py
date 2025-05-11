import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.1
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    USE_CORPORATE_NETWORK = True  # Set False for regular networks
    YAHOO_IMAP_IP = "98.136.96.91"  # Current Yahoo IMAP IP