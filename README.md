# RAG Q&A API with FastAPI

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on uploaded PDF documents. It uses Sentence Transformers for embeddings, FAISS for vector similarity search, a Hugging Face transformer model (like Llama 3.2) for generation, and FastAPI to serve the API.

It also includes optional features for storing documents in Azure Blob Storage and a bonus script for analyzing documents for outdated references and potential redundancy.

## Architecture Overview

1.  **Document Upload (`/upload` endpoint):**
    * A user uploads a PDF file via the API.
    * The FastAPI backend receives the file.
    * `PyPDF2` extracts text content from the PDF.
    * The text is split into smaller, potentially overlapping chunks.
    * `sentence-transformers` generates embeddings (vector representations) for each chunk.
    * The embeddings are added to a `FAISS` index for efficient similarity search.
    * The text chunks are stored, linked to their index positions.
    * The FAISS index and chunk data are persisted locally (using `faiss_index.idx` and `chunks_data.pkl`).

2.  **Querying (`/query` endpoint):**
    * A user sends a question to the API.
    * The question is embedded using the same `sentence-transformers` model.
    * `FAISS` searches the index for the chunks most similar (relevant) to the question embedding.
    * The text of these relevant chunks is retrieved to form the context.
    * A prompt is constructed containing the original question and the retrieved context, instructing the LLM on how to answer (e.g., use only the context, maintain a specific tone).
    * The prompt is sent to the Hugging Face `transformers` pipeline (using a pre-loaded LLM like `meta-llama/Llama-3.2-1B-Instruct`).
    * The LLM generates an answer based on the prompt and context.
    * The answer is returned to the user.

3.  **Azure Blob Storage (Optional Utility - `app/azure_blob_utils.py`):**
    * Provides functions to upload files (like the source PDFs) to an Azure Blob Storage container. This is primarily a utility and not directly used by the core `/upload` or `/query` endpoints in this version. Requires `AZURE_CONNECTION_STRING` to be set in `.env`.

4.  **Document Analyzer (Bonus Script/Endpoint - `app/document_analyzer.py`, `/analyze/{filename}`):**
    * A script or API endpoint that analyzes a given PDF (requires file access).
    * Detects potentially outdated content by looking for old dates/years.
    * Detects potentially redundant content by comparing the similarity of text chunks using embeddings.
    * Outputs findings to an Excel file (`document_analysis.xlsx`).

## Project Structure

rag_llm_project/├── app/                  # Main application code│   ├── init.py       # Makes 'app' a Python package│   ├── config.py         # Configuration variables (reads from .env)│   ├── azure_blob_utils.py # Azure Blob Storage utilities│   ├── rag_core.py       # Core RAG logic (PDF processing, embedding, FAISS, LLM query)│   ├── document_analyzer.py # Bonus document analysis script│   └── main.py           # FastAPI application definition and endpoints├── .env.example          # Example environment variables -> COPY TO .env AND FILL IN!├── .gitignore            # Files/directories to ignore in Git (includes .env)├── requirements.txt      # Python package dependencies├── README.md             # This file└── faiss_index.idx       # Saved FAISS index (created after running) - Consider gitignoring└── chunks_data.pkl       # Saved chunks list (created after running) - Consider gitignoringOptional:└── uploads/              # Example directory for storing original PDFs if needed by analysis endpoint
## Setup Instructions

1.  **Clone the Repository (or create files):**
    * If cloning: `git clone <your-github-repo-url>`
    * If creating manually: Create the directory structure and files as shown above, copying the content for each file.
    * Navigate into the project directory: `cd rag_llm_project`

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* `faiss-cpu` is listed. If you have a CUDA-enabled GPU and want to use it, install `faiss-gpu` instead (requires CUDA toolkit installed). Check FAISS documentation for specifics.*

4.  **Configure Environment Variables (CRITICAL STEP):**
    * **Copy `.env.example` to `.env`:**
        ```bash
        # Linux/macOS
        cp .env.example .env

        # Windows
        copy .env.example .env
        ```
    * **Edit the `.env` file** with a text editor.
    * **Replace the placeholder values** with your actual secrets:
        * `HF_TOKEN`: Your Hugging Face Hub token (needed for gated models).
        * `AZURE_CONNECTION_STRING`: Your full Azure Storage connection string (needed for Azure uploads).
    * **IMPORTANT:** The `.gitignore` file is already configured to ignore `.env`, so you won't accidentally commit your secrets. **Never commit your `.env` file.**

5.  **Run the FastAPI Application:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    * `--reload`: Automatically restarts the server when code changes (useful for development).
    * `--host 0.0.0.0`: Makes the server accessible on your network (use `127.0.0.1` for local access only).
    * `--port 8000`: Specifies the port to run on.

6.  **Access the API:**
    * Open your browser and navigate to `http://localhost:8000/docs` (or `http://<your-ip>:8000/docs` if using `0.0.0.0`). This will show the interactive Swagger UI for testing the endpoints.

## Usage

1.  **Upload Documents:**
    * Use the `/docs` UI or a tool like `curl` to send a POST request to `/upload`.
    * Attach the PDF file you want to index.
    * Example using `curl`:
        ```bash
        curl -X POST "http://localhost:8000/upload" \
             -H "accept: application/json" \
             -H "Content-Type: multipart/form-data" \
             -F "file=@/path/to/your/document.pdf;type=application/pdf"
        ```

2.  **Ask Questions:**
    * Use the `/docs` UI or `curl` to send a POST request to `/query`.
    * Provide the question as a query parameter.
    * Example using `curl`:
        ```bash
        curl -X POST "http://localhost:8000/query?question=What%20is%20the%20policy%20on%20dress%20code%3F" \
             -H "accept: application/json"
        ```
        (Replace the question with your URL-encoded query).

3.  **Run Document Analyzer (Optional):**
    * **Via Script:**
        * Ensure the PDF you want to analyze is accessible by the server/script.
        * Run the script from the project root directory (`rag_llm_project/`):
            ```bash
            python -m app.document_analyzer path/to/your/document.pdf
            ```
        * Check for the output file (e.g., `document_analysis.xlsx`) in the project root.
    * **Via API Endpoint (`/analyze/{filename}`):**
        * **Requires setup:** You need to ensure the PDF file corresponding to `{filename}` is stored where the server can access it (e.g., in an `uploads/` directory) and potentially modify the path logic in `app/main.py`.
        * Send a POST request (e.g., via `/docs` UI or `curl`):
            ```bash
            curl -X POST "http://localhost:8000/analyze/your_document.pdf" \
                 -H "accept: application/json"
            ```

## Notes

* **Persistence:** The FAISS index (`faiss_index.idx`) and chunks list (`chunks_data.pkl`) are saved in the directory where you run `uvicorn`. If you restart the server, it will load the existing index. Consider adding these files to `.gitignore` if you don't want to commit the indexed data.
* **Model Loading:** The embedding and LLM models are loaded into memory on application startup. This might take some time and require significant RAM/VRAM depending on the models chosen.
* **Error Handling:** Basic error handling is included, but production applications may require more robust error management and monitoring.
