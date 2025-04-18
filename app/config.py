# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
# This allows you to keep sensitive information out of the code
load_dotenv()

# --- Azure Configuration ---
# Reads the connection string from the environment variable 'AZURE_CONNECTION_STRING'
# Provide a default placeholder if the environment variable is not set (optional, but helps avoid errors)
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "YOUR_AZURE_CONNECTION_STRING_PLACEHOLDER")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "pdf-container") # Default container name

# --- Model Configuration ---
# Reads the Hugging Face token from the environment variable 'HF_TOKEN'
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_PLACEHOLDER")
# Define the embedding model
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Define the primary LLM model ID
LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# LLM_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ" # Alternative model

# --- RAG Configuration ---
CHUNK_SIZE = 500 # Number of words per text chunk
CHUNK_OVERLAP = 50 # Number of words overlapping between consecutive chunks
TOP_K_RESULTS = 3 # Number of relevant chunks to retrieve for context

# --- File Paths for Persistence ---
# Defines where the FAISS index and chunks data will be saved locally
# These files will be created in the same directory where the app is run from
FAISS_INDEX_PATH = "faiss_index.idx"
CHUNKS_DATA_PATH = "chunks_data.pkl"

# --- Document Analyzer Config ---
OUTDATED_THRESHOLD_YEARS = 5 # Threshold for considering a year reference as potentially outdated
REDUNDANCY_SIMILARITY_THRESHOLD = 0.95 # Cosine similarity threshold for marking chunks as potentially redundant
ANALYSIS_OUTPUT_FILE = "document_analysis.xlsx" # Default filename for the analysis report

# --- Helper function to check if essential configs are set ---
def check_essential_configs():
    """Checks if essential configuration variables (secrets) seem to be set."""
    essential_missing = False
    if not HF_TOKEN or "PLACEHOLDER" in HF_TOKEN:
        print("Warning: HF_TOKEN is not set or using placeholder in environment variables / .env file.")
        essential_missing = True
    if not AZURE_CONNECTION_STRING or "PLACEHOLDER" in AZURE_CONNECTION_STRING:
        # This might be optional if Azure functionality isn't used, but warn anyway
        print("Warning: AZURE_CONNECTION_STRING is not set or using placeholder in environment variables / .env file.")
        # essential_missing = True # Decide if Azure is strictly required for startup
    return not essential_missing

# You can optionally call check_essential_configs() when the app starts
# if needed, but it's often better handled where the config is used.
