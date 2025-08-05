import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = 'AIzaSyAIXAPHNDHZJILiRB04Sk-FGEPEyYm1Ydg'
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# File Configuration
SUPPORTED_FORMATS = [".txt", ".md", ".pdf"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 1200  # Larger chunks to keep related info together
CHUNK_OVERLAP = 600  # Large overlap to ensure context continuity

# Vector DB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "project_documents"

# Model Configuration
GEMINI_MODEL = "gemini-2.5-flash"

# Multimodal Configuration
ENABLE_MULTIMODAL = True
MAX_VISUAL_ELEMENTS_PER_DOC = 20  # Limit visual elements to prevent memory issues
IMAGE_ANALYSIS_MAX_TOKENS = 300  # Limit AI analysis length
