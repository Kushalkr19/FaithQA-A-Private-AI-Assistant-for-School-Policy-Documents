# app/rag_core.py
import faiss
import numpy as np
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from functools import lru_cache
import os
import pickle
import logging
import time

# Use relative import to access config variables
from . import config

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Global Variables / State Management ---
# These variables hold the loaded models and data.
# In a production scenario, consider a more robust state management class.
embedder = None
index = None
chunks_list = [] # Stores the actual text content of the chunks

# --- Initialization Functions ---

def initialize_embedder():
    """Loads the Sentence Transformer model specified in the config."""
    global embedder
    # Load only if it hasn't been loaded yet
    if embedder is None:
        model_name = config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {model_name}...")
        start_time = time.time()
        try:
            embedder = SentenceTransformer(model_name)
            elapsed_time = time.time() - start_time
            logger.info(f"Embedding model loaded successfully in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
            # Depending on criticality, you might raise the exception or handle it
            # raise e # Or return None / set embedder to None
    return embedder

def initialize_faiss_index(embedding_dim):
    """
    Initializes a new FAISS index or loads an existing one from the path specified in config.
    Args:
        embedding_dim (int): The dimension of the embeddings (required to initialize a new index).
    Returns:
        faiss.Index | None: The loaded or initialized FAISS index, or None on failure.
    """
    global index
    index_path = config.FAISS_INDEX_PATH
    # Load only if not already loaded
    if index is None:
        if os.path.exists(index_path):
            logger.info(f"Attempting to load FAISS index from {index_path}...")
            try:
                index = faiss.read_index(index_path)
                logger.info(f"FAISS index loaded successfully with {index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading FAISS index from {index_path}: {e}. A new index will be created.", exc_info=True)
                # Fallback to creating a new index if loading fails
                index = faiss.IndexFlatL2(embedding_dim)
                logger.info(f"Initialized new empty FAISS index (dim={embedding_dim}).")
        else:
            logger.info(f"FAISS index file not found at {index_path}. Initializing new empty index (dim={embedding_dim}).")
            index = faiss.IndexFlatL2(embedding_dim)
    return index

def load_chunks():
    """Loads the list of text chunks from the path specified in config."""
    global chunks_list
    chunks_path = config.CHUNKS_DATA_PATH
    # Load only if the list is currently empty
    if not chunks_list: # Check if the list is empty
        if os.path.exists(chunks_path):
            logger.info(f"Attempting to load chunks data from {chunks_path}...")
            try:
                with open(chunks_path, 'rb') as f:
                    chunks_list = pickle.load(f)
                logger.info(f"Loaded {len(chunks_list)} chunks successfully.")
            except Exception as e:
                logger.error(f"Error loading chunks data from {chunks_path}: {e}. Starting with an empty chunks list.", exc_info=True)
                chunks_list = [] # Ensure it's an empty list on failure
        else:
            logger.info(f"Chunks data file not found at {chunks_path}. Starting with an empty chunks list.")
            chunks_list = []
    return chunks_list

def initialize_rag_components():
    """Initializes all necessary RAG components: embedder, FAISS index, and chunks list."""
    logger.info("Initializing RAG components...")
    start_time = time.time()

    # 1. Initialize Embedder
    current_embedder = initialize_embedder()
    if current_embedder is None:
        logger.error("Failed to initialize embedder. RAG pipeline may not function.")
        return # Cannot proceed without embedder

    embedding_dim = current_embedder.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")

    # 2. Initialize FAISS Index
    current_index = initialize_faiss_index(embedding_dim)
    if current_index is None:
         logger.error("Failed to initialize FAISS index. RAG pipeline may not function.")
         return # Cannot proceed without index

    # 3. Load Chunks
    current_chunks = load_chunks()

    # 4. Consistency Check (Important!)
    if current_index.ntotal != len(current_chunks):
        logger.warning(f"FAISS index size ({current_index.ntotal}) does not match loaded chunks count ({len(current_chunks)}). "
                       "This indicates inconsistency, possibly from a previous error. Resetting index and chunks list.")
        # Resetting state to prevent errors during querying
        global index, chunks_list
        index = faiss.IndexFlatL2(embedding_dim) # Create new empty index
        chunks_list = [] # Reset chunks list
        save_rag_state() # Attempt to save the clean state immediately

    elapsed_time = time.time() - start_time
    logger.info(f"RAG components initialization finished in {elapsed_time:.2f} seconds. Index size: {index.ntotal}, Chunks loaded: {len(chunks_list)}")

# --- Persistence Function ---

def save_rag_state():
    """Saves the current state of the FAISS index and the chunks list to files."""
    global index, chunks_list
    index_path = config.FAISS_INDEX_PATH
    chunks_path = config.CHUNKS_DATA_PATH
    state_saved = True

    # Save FAISS index
    if index is not None:
        try:
            logger.info(f"Saving FAISS index ({index.ntotal} vectors) to {index_path}...")
            faiss.write_index(index, index_path)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {index_path}: {e}", exc_info=True)
            state_saved = False

    # Save chunks list using pickle
    try:
        logger.info(f"Saving chunks list ({len(chunks_list)} chunks) to {chunks_path}...")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks_list, f)
        logger.info("Chunks list saved successfully.")
    except Exception as e:
        logger.error(f"Error saving chunks data to {chunks_path}: {e}", exc_info=True)
        state_saved = False

    return state_saved

# --- PDF Processing and Indexing Functions ---

def extract_text_from_pdf(file_path):
    """
    Extracts text content from all pages of a given PDF file.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str | None: The extracted text concatenated from all pages, or None if an error occurs.
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    try:
        reader = PdfReader(file_path)
        if not reader.pages:
             logger.warning(f"No pages found in PDF: {file_path}")
             return "" # Return empty string if no pages
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        logger.info(f"Successfully extracted text from {len(reader.pages)} pages.")
        return text
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}", exc_info=True)
        return None

def split_text_into_chunks(text):
    """
    Splits a given text into overlapping chunks based on word count, using settings from config.
    Args:
        text (str): The text content to be split.
    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        logger.warning("Attempted to split empty text into chunks.")
        return []

    words = text.split() # Split by whitespace
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    step = chunk_size - overlap

    if step <= 0:
        logger.error(f"Chunk size ({chunk_size}) must be greater than chunk overlap ({overlap}). Adjust config.")
        # Handle error: maybe default to non-overlapping chunks or raise exception
        step = chunk_size # Fallback to non-overlapping

    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks

def add_document_to_index(file_path):
    """
    Processes a PDF document: extracts text, creates chunks, generates embeddings,
    adds them to the FAISS index and chunks list, and saves the updated state.
    Args:
        file_path (str): Path to the PDF document to be added.
    Returns:
        int: The number of new chunks successfully added to the index, or 0 if processing fails.
    """
    global index, chunks_list, embedder
    if embedder is None or index is None:
        logger.error("Cannot add document: RAG components (embedder or index) are not initialized.")
        return 0

    # 1. Extract text
    text = extract_text_from_pdf(file_path)
    if text is None: # Check for None explicitly, as empty string is valid but won't yield chunks
        logger.error(f"Failed to extract text from {file_path}. Document not added.")
        return 0
    if not text.strip():
         logger.warning(f"Extracted text from {file_path} is empty or whitespace only. No chunks to add.")
         return 0

    # 2. Create chunks
    new_chunks = split_text_into_chunks(text)
    if not new_chunks:
        logger.warning(f"No chunks were generated from {file_path}. Document not added.")
        return 0

    # 3. Generate embeddings
    logger.info(f"Generating embeddings for {len(new_chunks)} new chunks from {os.path.basename(file_path)}...")
    try:
        start_time = time.time()
        # Use show_progress_bar=True for visual feedback in interactive environments if desired
        new_embeddings = embedder.encode(new_chunks, show_progress_bar=False)
        embeddings_np = np.array(new_embeddings, dtype='float32') # FAISS requires float32
        elapsed_time = time.time() - start_time
        logger.info(f"Embeddings generated in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to generate embeddings for chunks from {file_path}: {e}", exc_info=True)
        return 0

    # 4. Add to FAISS index and chunks list
    try:
        index.add(embeddings_np)
        chunks_list.extend(new_chunks) # Add the corresponding text chunks
        logger.info(f"Successfully added {len(new_chunks)} chunks to index. New total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Failed to add embeddings/chunks to index/list: {e}", exc_info=True)
        # Attempt to rollback? Difficult with FAISS in-memory add. Best to log and potentially re-initialize later.
        return 0 # Indicate failure

    # 5. Save the updated state
    if not save_rag_state():
        logger.warning("Failed to save RAG state after adding document. State might be inconsistent on restart.")
        # Decide if this is critical enough to return 0

    return len(new_chunks)

# --- LLM Loading and Querying Functions ---

@lru_cache(maxsize=1) # Cache the loaded pipeline object for efficiency
def get_llm_generator():
    """
    Loads the Hugging Face text generation pipeline using the model specified in config.
    Uses LRU cache to load the model only once.
    Returns:
        transformers.Pipeline | None: The loaded text generation pipeline, or None on failure.
    """
    model_id = config.LLM_MODEL_ID
    hf_token = config.HF_TOKEN
    logger.info(f"Attempting to load LLM model: {model_id}...")

    # Validate token presence (basic check)
    use_auth = bool(hf_token and "PLACEHOLDER" not in hf_token)
    if not use_auth:
         logger.warning(f"Hugging Face token (HF_TOKEN) not found or is placeholder. "
                        f"Loading '{model_id}' without authentication. This may fail for gated models.")

    try:
        start_time = time.time()
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_token if use_auth else None,
            trust_remote_code=True # Required by some models like Mistral-AWQ
        )
        logger.info("Tokenizer loaded.")

        # Determine torch dtype based on CUDA availability for efficiency
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Using torch dtype: {dtype} (CUDA available: {torch.cuda.is_available()})")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=hf_token if use_auth else None,
            torch_dtype=dtype,
            device_map="auto", # Automatically use GPU if available, otherwise CPU
            trust_remote_code=True # Required by some models
        )
        elapsed_time = time.time() - start_time
        logger.info(f"LLM model '{model_id}' loaded successfully in {elapsed_time:.2f} seconds.")

        # Create and return the pipeline
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
            # device_map is handled by model loading
        )
    except ImportError as e:
         logger.error(f"ImportError during LLM loading: {e}. Ensure all 'transformers' dependencies (like accelerate) are installed.", exc_info=True)
         return None
    except OSError as e:
         # Handle specific errors like model not found or connection issues
         logger.error(f"OSError during LLM loading (model not found or connection issue?): {e}", exc_info=True)
         if "authentication" in str(e).lower():
              logger.error("Hint: Check your HF_TOKEN and ensure you have accepted the model's terms on the Hugging Face Hub.")
         return None
    except Exception as e:
        # Catch any other unexpected errors during loading
        logger.error(f"An unexpected error occurred loading the LLM model or tokenizer: {e}", exc_info=True)
        return None

def build_prompt(question, context):
    """
    Constructs the instruction-following prompt for the LLM, incorporating the retrieved context.
    Args:
        question (str): The user's original question.
        context (str): The relevant text chunks retrieved from the document index.
    Returns:
        str: The fully formatted prompt string.
    """
    # This prompt template is based on the one used in the user's notebook.
    # It provides instructions, context, and the question.
    prompt = (
        "### Instruction:\n"
        "You are a helpful assistant for a Christian school. "
        "Answer the user's question using only the provided document context. "
        "Maintain a respectful, compassionate tone, and reflect values such as ethics, honesty, integrity, and kindness. "
        "If the question includes inappropriate or offensive content, kindly say 'I am sorry, this is inappropriate, please ask something else.' "
        "For out-of-scope questions where the context does not provide an answer, respond politely with 'Sorry, I can't provide an answer to this as it's outside the scope of the document.'\n"
        "Do not make up information. Avoid content that could be offensive or discriminatory.\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:" # The model should generate text following this marker
    )
    return prompt

def query_rag_pipeline(question):
    """
    Executes the full RAG pipeline for a given question:
    1. Embeds the question.
    2. Searches the FAISS index for relevant chunks.
    3. Retrieves the corresponding text chunks.
    4. Builds a prompt with the question and context.
    5. Generates an answer using the LLM.
    6. Performs basic post-processing on the answer.

    Args:
        question (str): The user's question.

    Returns:
        str: The generated answer, or an error/fallback message.
    """
    global index, chunks_list, embedder
    start_time = time.time()

    # --- Pre-computation Checks ---
    if index is None or embedder is None:
        logger.error("Query failed: RAG components (index or embedder) not initialized.")
        return "Sorry, the document query system is not ready. Please try again later."
    if index.ntotal == 0:
        logger.warning("Query attempted on an empty index. No documents have been processed yet.")
        return "Sorry, no documents have been loaded yet. Please upload a document first."

    generator = get_llm_generator()
    if generator is None:
        logger.error("Query failed: LLM generator could not be loaded.")
        return "Sorry, the language model component is unavailable. Please try again later."

    logger.info(f"Processing query: '{question}'")

    try:
        # --- 1. Embed the Question ---
        logger.debug("Embedding the question...")
        q_vector = embedder.encode([question]).astype('float32')
        logger.debug("Question embedded.")

        # --- 2. Search FAISS Index ---
        k = min(config.TOP_K_RESULTS, index.ntotal) # Ensure k is not larger than the number of items in index
        logger.debug(f"Searching FAISS index for top {k} relevant chunks...")
        distances, indices = index.search(q_vector, k=k)
        logger.debug(f"FAISS search completed. Indices found: {indices}")

        # Check if any relevant chunks were found
        if not indices.size or indices[0][0] == -1: # FAISS returns -1 if no neighbors found within radius (for IndexIVF etc.) or if k=0
             logger.warning(f"No relevant chunks found in the index for the question: '{question}'")
             # Return a specific message indicating lack of relevant context
             return "Sorry, I couldn't find information relevant to your question in the loaded documents."

        # --- 3. Retrieve Text Chunks ---
        logger.debug("Retrieving text chunks based on indices...")
        # Ensure indices are valid and within the bounds of the chunks_list
        retrieved_chunks = [chunks_list[idx] for idx in indices[0] if 0 <= idx < len(chunks_list)]
        if not retrieved_chunks:
             logger.warning(f"FAISS returned indices {indices[0]}, but failed to retrieve corresponding chunks. List length: {len(chunks_list)}")
             return "Sorry, an internal error occurred while retrieving document context."

        context = "\n\n---\n\n".join(retrieved_chunks) # Join chunks with a separator
        logger.debug(f"Context constructed ({len(context)} chars). Start: {context[:200]}...")

        # --- 4. Build the Prompt ---
        prompt = build_prompt(question, context)
        logger.debug(f"Prompt built. Length: {len(prompt)}")
        # logger.debug(f"Full Prompt:\n{prompt}") # Uncomment for debugging prompts

        # --- 5. Generate Answer with LLM ---
        logger.info("Sending prompt to LLM for answer generation...")
        generation_start_time = time.time()
        # Define generation parameters
        # max_new_tokens: Controls the maximum length of the generated answer.
        # num_return_sequences: We typically only need one answer.
        # do_sample=False: For more deterministic, factual answers based on context. Set to True for more creative/varied answers.
        # temperature, top_p, top_k: Only relevant if do_sample=True. Control randomness.
        result = generator(
            prompt,
            max_new_tokens=300, # Adjust as needed
            num_return_sequences=1,
            do_sample=False,
            # Add other parameters like temperature=0.7, top_k=50 if do_sample=True
        )
        generation_elapsed_time = time.time() - generation_start_time
        logger.info(f"LLM generation completed in {generation_elapsed_time:.2f} seconds.")

        # --- 6. Post-process Answer ---
        raw_answer = result[0]["generated_text"]
        # Extract only the text generated *after* the "### Answer:" marker in the prompt
        answer = raw_answer.split("### Answer:")[-1].strip()

        logger.debug(f"Raw answer from LLM: {answer}")

        # Basic checks for common failure modes or out-of-scope responses from the LLM
        # These checks might need refinement based on observed LLM behavior
        fallback_message = "Sorry, I can't provide an answer to this as it seems to be outside the scope of the document or the information is not available."
        if not answer or \
           answer.startswith("###") or \
           len(answer) < 10 or \
           'sorry, i can\'t provide an answer' in answer.lower() or \
           'outside the scope of the document' in answer.lower() or \
           'i don’t know' in answer.lower() or \
           'i am sorry, this is inappropriate' in answer.lower():
             logger.warning(f"LLM response might be out-of-scope, empty, or invalid. Using fallback message. Original answer: '{answer}'")
             answer = fallback_message

        total_elapsed_time = time.time() - start_time
        logger.info(f"Query processed successfully in {total_elapsed_time:.2f} seconds. Returning answer.")
        return answer

    except Exception as e:
        # Catch-all for any unexpected errors during the query process
        logger.error(f"An unexpected error occurred during the RAG query pipeline for question '{question}': {e}", exc_info=True)
        return "Sorry, an internal error occurred while processing your question. Please try again later."

# --- Initialize components when the module is first imported ---
# This ensures models are loaded when the application starts (e.g., when FastAPI imports it)
initialize_rag_components()
