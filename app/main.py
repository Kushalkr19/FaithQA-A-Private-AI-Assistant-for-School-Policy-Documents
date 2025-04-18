# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
import os
import tempfile
import logging
import time
from contextlib import asynccontextmanager

# Import necessary functions and components from our RAG core module
# Using relative imports as they are part of the same 'app' package
from . import rag_core
from . import config # May need config values directly here too

# --- Logging Setup ---
# Configure logging for the application
logging.basicConfig(
    level=logging.INFO, # Set the default logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Get a logger instance specific to this module
logger = logging.getLogger(__name__)

# --- Application Lifespan Management (for Startup/Shutdown) ---
# This is the modern way in FastAPI (>= 0.90) to handle startup/shutdown events
# replacing @app.on_event("startup") / @app.on_event("shutdown")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs ON STARTUP
    logger.info("Application startup sequence initiated...")
    start_time = time.time()
    # Initialize RAG components. rag_core.initialize_rag_components() is called
    # automatically when rag_core is imported, but we can log here.
    # We could add checks here, e.g., ensure models are loaded.
    if rag_core.embedder is None or rag_core.index is None or rag_core.get_llm_generator() is None:
         logger.critical("CRITICAL: RAG components failed to initialize properly during module import. API might not function correctly.")
         # Depending on severity, you might prevent the app from fully starting,
         # but for now, we just log critically.
    else:
         logger.info("RAG components appear to be initialized.")

    # Check if essential configurations (like secrets) are set
    if not config.check_essential_configs():
         logger.warning("Essential configurations (HF_TOKEN, AZURE_CONNECTION_STRING) might be missing or using placeholders. Check .env file.")

    end_time = time.time()
    logger.info(f"Application startup sequence completed in {end_time - start_time:.2f} seconds.")

    yield # The application runs while yielded

    # Code here runs ON SHUTDOWN
    logger.info("Application shutdown sequence initiated...")
    # Perform cleanup tasks if necessary
    # For example, explicitly saving state (though rag_core saves on update)
    # rag_core.save_rag_state() # Optional: ensure state is saved on graceful shutdown
    logger.info("Application shutdown sequence completed.")


# --- FastAPI Application Instance ---
# Create the FastAPI app instance, including the lifespan context manager
app = FastAPI(
    title="RAG Q&A API",
    description="API for uploading PDF documents and asking questions using Retrieval-Augmented Generation.",
    version="1.0.1", # Increment version
    lifespan=lifespan # Register the lifespan manager
)

# --- Dependency for RAG Readiness ---
# Optional: Create a dependency that checks if RAG components are ready before processing requests
async def check_rag_readiness():
    """Dependency to ensure RAG components are initialized."""
    if rag_core.embedder is None or rag_core.index is None or rag_core.get_llm_generator() is None:
        logger.error("Dependency Check Failed: RAG components not ready.")
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG components are not initialized.")
    # Could add more checks, e.g., index.ntotal > 0 for query endpoint
    # logger.debug("Dependency Check Passed: RAG components ready.") # Optional debug log

# --- API Endpoints ---

@app.get("/health", tags=["Status"])
async def health_check():
    """
    Provides a basic health check endpoint to confirm the API is running.
    """
    logger.info("Health check endpoint called.")
    # More advanced checks could be added here (e.g., check model status, DB connection)
    return {"status": "alive", "message": "API is running."}

@app.post("/upload", tags=["Documents"], dependencies=[Depends(check_rag_readiness)])
async def upload_document(
    file: UploadFile = File(..., description="The PDF document to be processed and indexed.")
):
    """
    Handles the upload of a PDF file. The file content is extracted, chunked,
    embedded, and added to the vector index.
    """
    filename = file.filename
    logger.info(f"Received request to upload file: {filename}")

    # Validate file type (ensure it's a PDF)
    if not filename or not filename.lower().endswith(".pdf"):
        logger.warning(f"Upload rejected: File '{filename}' is not a PDF.")
        raise HTTPException(
            status_code=400, # Bad Request
            detail="Invalid file type. Only PDF (.pdf) files are accepted."
        )

    # Create a temporary file to safely store the uploaded content
    try:
        # Using NamedTemporaryFile ensures the file is automatically cleaned up
        # even if errors occur (unless delete=False is used and cleanup fails)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Read the uploaded file content asynchronously
            content = await file.read()
            # Write the content to the temporary file
            tmp_file.write(content)
            tmp_path = tmp_file.name # Get the path to the temporary file
            logger.info(f"File '{filename}' saved temporarily to: {tmp_path}")

        # Process the document using the function from rag_core
        # This handles text extraction, chunking, embedding, indexing, and saving state
        logger.info(f"Processing temporary file: {tmp_path}")
        chunks_added = rag_core.add_document_to_index(tmp_path)
        logger.info(f"Processing complete for temporary file: {tmp_path}")

    except HTTPException as http_exc:
        # Re-raise exceptions that are already FastAPI HTTPExceptions
        logger.error(f"HTTPException during upload of {filename}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during file handling or processing
        logger.error(f"An unexpected error occurred processing file {filename}: {e}", exc_info=True)
        # Raise a generic server error response
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected error occurred while processing the document: {str(e)}"
        )
    finally:
        # Ensure the temporary file is deleted if it still exists
        # This is crucial if delete=False was used or if an error occurred before the 'with' block exited
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Temporary file '{tmp_path}' removed successfully.")
            except Exception as cleanup_e:
                # Log if cleanup fails, but don't let it mask the original error
                logger.error(f"Error removing temporary file '{tmp_path}': {cleanup_e}", exc_info=True)

    # Check the result of adding the document
    if chunks_added > 0:
        logger.info(f"Successfully added {chunks_added} chunks from '{filename}'. Total indexed chunks: {rag_core.index.ntotal}")
        return {
            "message": f"Document '{filename}' processed successfully.",
            "filename": filename,
            "chunks_added": chunks_added,
            "total_indexed_chunks": rag_core.index.ntotal
        }
    else:
        # This could happen if the PDF was empty, unreadable, or text extraction failed silently in rag_core
        logger.warning(f"File '{filename}' processed, but no new chunks were added to the index.")
        # Return a success message but indicate no chunks were added.
        # Alternatively, could return a different status code (e.g., 202 Accepted but not fully processed)
        return {
            "message": f"Document '{filename}' processed, but no new content was added (possibly empty or unreadable).",
            "filename": filename,
            "chunks_added": 0,
            "total_indexed_chunks": rag_core.index.ntotal
        }


@app.post("/query", tags=["Query"], dependencies=[Depends(check_rag_readiness)])
async def query_document(
    question: str = Query(..., min_length=3, description="The question to ask about the indexed documents.")
):
    """
    Receives a question, processes it using the RAG pipeline (embedding, searching,
    context retrieval, LLM generation), and returns the generated answer.
    """
    logger.info(f"Received query: '{question}'")

    # Basic validation (redundant due to Query(..., min_length=3) but good practice)
    if not question or len(question) < 3:
        logger.warning("Query rejected: Question is too short or empty.")
        raise HTTPException(
            status_code=400, # Bad Request
            detail="Query failed: Question must be at least 3 characters long."
        )

    # Check if the index is populated before querying
    if rag_core.index is None or rag_core.index.ntotal == 0:
         logger.warning("Query attempted but no documents are indexed.")
         raise HTTPException(
              status_code=404, # Not Found (or 400 Bad Request)
              detail="Cannot process query: No documents have been successfully indexed yet. Please upload documents first."
         )

    try:
        # Call the main RAG query function from rag_core
        start_time = time.time()
        answer = rag_core.query_rag_pipeline(question)
        end_time = time.time()
        logger.info(f"Query processed in {end_time - start_time:.2f} seconds. Answer length: {len(answer)}")
        # Return the question and the generated answer
        return {"question": question, "answer": answer}

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        logger.error(f"HTTPException during query processing for '{question}': {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Handle any unexpected errors during the query process
        logger.error(f"An unexpected error occurred processing query '{question}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected error occurred while processing your query: {str(e)}"
        )


# --- Optional: Endpoint to trigger document analysis ---
# Note: This endpoint requires careful consideration of file access and security.
# The current implementation assumes files are accessible via a simple path, which
# might not be suitable for all deployment scenarios.
@app.post("/analyze/{filename}", tags=["Analysis"], dependencies=[Depends(check_rag_readiness)])
async def trigger_analysis(filename: str):
    """
    (Optional/Example) Triggers the document analysis script for a specified filename.
    Requires the file to be accessible by the server at a predefined location.
    **WARNING:** File path construction needs careful implementation based on deployment strategy.
    """
    from . import document_analyzer # Import locally to avoid loading if not used

    # --- !!! SECURITY WARNING & IMPLEMENTATION NOTE !!! ---
    # The following path construction is a **highly simplified example**.
    # In a real application, you MUST determine how the server accesses
    # the original PDF files (e.g., from a shared volume, object storage,
    # or an internal 'uploads' directory managed by the app).
    # Directly using user-provided filenames in paths is dangerous (path traversal).
    # Consider using file IDs or a secure mapping instead.
    # For this example, we assume an 'uploads' directory exists at the project root.
    upload_dir = "uploads" # Example: Define a directory relative to where the app runs
    # Securely join paths and potentially sanitize filename
    # os.path.basename ensures only the filename part is used, preventing traversal like ../../
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(upload_dir, safe_filename)
    # --- !!! END WARNING !!! ---

    logger.info(f"Request received to analyze document: {safe_filename} (expected at path: {file_path})")

    # Check if the constructed file path actually exists
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.error(f"Analysis failed: File not found or is not a file at expected path: {file_path}")
        raise HTTPException(status_code=404, detail=f"File '{safe_filename}' not found for analysis at expected location.")

    try:
        # Run the analysis function from the document_analyzer module
        start_time = time.time()
        output_report_path = document_analyzer.analyze_document(file_path)
        end_time = time.time()
        logger.info(f"Analysis for '{safe_filename}' completed in {end_time - start_time:.2f} seconds.")

        # Check if the analysis produced an output file
        if output_report_path:
            # In a real app, you might return the report file (e.g., using FileResponse)
            # or provide a link if it's stored publicly.
            # For simplicity, just return a message indicating success and location.
            return {
                "message": f"Analysis complete for {safe_filename}.",
                "report_generated": True,
                "report_location_info": f"Report saved on server at '{output_report_path}'." # Note: Path is relative to server
            }
        else:
            # Analysis ran but found no issues or failed to save the report
            return {
                 "message": f"Analysis ran for {safe_filename}, but no major issues were found or the report could not be saved.",
                 "report_generated": False
             }
    except Exception as e:
        # Handle errors during the analysis process
        logger.error(f"Error during analysis of {safe_filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected error occurred during document analysis: {str(e)}"
        )

# --- How to Run (for documentation purposes) ---
# Use uvicorn to run the application:
# Example: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# - `app.main`: Refers to the 'app' directory (package) and the 'main.py' file.
# - `app`: Refers to the FastAPI instance created as `app = FastAPI(...)` within main.py.
# - `--reload`: Enables auto-reload on code changes (for development).
# - `--host 0.0.0.0`: Makes the server accessible on the network.
# - `--port 8000`: Specifies the port number.
