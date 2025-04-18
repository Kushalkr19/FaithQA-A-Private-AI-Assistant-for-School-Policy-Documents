# app/document_analyzer.py
import re
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging

# Use relative import for config
from . import config
# Import necessary functions from rag_core, potentially initializing components if not already done
# Note: Running this as a standalone script might trigger RAG component initialization
from .rag_core import extract_text_from_pdf, split_text_into_chunks, initialize_embedder

# Setup logger for this module
logger = logging.getLogger(__name__)

def find_outdated_references(text, threshold_years=config.OUTDATED_THRESHOLD_YEARS):
    """
    Scans text for four-digit year mentions and flags those older than a specified threshold.

    Args:
        text (str): The text content to scan.
        threshold_years (int): The age in years beyond which a reference is considered potentially outdated.

    Returns:
        list[dict]: A list of dictionaries, each representing an outdated reference found.
                    Returns an empty list if no outdated references are found.
    """
    if not text:
        logger.warning("Cannot find outdated references in empty text.")
        return []

    outdated_issues = []
    current_year = datetime.now().year
    # Calculate the cutoff year (exclusive, e.g., if threshold=5, current=2025, cutoff=2020, years < 2020 are flagged)
    cutoff_year = current_year - threshold_years
    logger.info(f"Scanning for potentially outdated references (years before {cutoff_year})...")

    # Regex to find plausible 4-digit years (e.g., 19xx, 20xx)
    # Adjust range if needed (e.g., exclude future years, start from a specific past year)
    year_pattern = r'\b(19[89]\d|20\d{2})\b' # Example: finds years from 1980 to 2099

    for match in re.finditer(year_pattern, text):
        try:
            year = int(match.group())
            # Check if the found year is before the calculated cutoff year
            if year < cutoff_year:
                # Capture a snippet of text around the found year for context
                context_window = 40 # Number of characters before and after the year
                start_index = max(0, match.start() - context_window)
                end_index = min(len(text), match.end() + context_window)
                # Get the snippet and replace newlines with spaces for better readability in output
                snippet = text[start_index:end_index].replace('\n', ' ')

                # Create a dictionary representing the issue found
                issue = {
                    "Issue Type": "Potentially Outdated Reference",
                    "Details": f"Mention of year {year} (>{threshold_years} years ago)",
                    "Excerpt": f"...{snippet}..." # Add ellipsis to indicate it's a snippet
                }
                outdated_issues.append(issue)
        except ValueError:
            # This should not happen with the regex, but included for safety
            logger.warning(f"Could not convert matched pattern '{match.group()}' to year.")
            continue

    logger.info(f"Found {len(outdated_issues)} potentially outdated references.")
    return outdated_issues

def find_redundant_content(chunks, embedder, similarity_threshold=config.REDUNDANCY_SIMILARITY_THRESHOLD):
    """
    Identifies potentially redundant content by calculating cosine similarity between chunk embeddings.
    Flags pairs of chunks with similarity exceeding the specified threshold.

    Args:
        chunks (list[str]): The list of text chunks.
        embedder (SentenceTransformer): The loaded sentence transformer model instance.
        similarity_threshold (float): The cosine similarity score above which chunks are considered redundant.

    Returns:
        list[dict]: A list of dictionaries, each representing a pair of potentially redundant chunks.
                    Returns an empty list if no redundancy is found or if input is insufficient.
    """
    if not chunks or len(chunks) < 2:
        logger.warning("Need at least two chunks to check for redundancy.")
        return []
    if embedder is None:
        logger.error("Embedder not available. Cannot perform redundancy check.")
        return []

    logger.info(f"Checking for redundancy among {len(chunks)} chunks (Similarity Threshold: {similarity_threshold})...")

    # 1. Generate embeddings for all chunks
    try:
        logger.debug("Generating embeddings for redundancy check...")
        embeddings = embedder.encode(chunks, show_progress_bar=False) # Set show_progress_bar=True for visual feedback
        embeddings_np = np.array(embeddings, dtype='float32')
        logger.debug("Embeddings generated.")
    except Exception as e:
        logger.error(f"Error generating embeddings during redundancy check: {e}", exc_info=True)
        return [] # Cannot proceed without embeddings

    # 2. Calculate Cosine Similarity Matrix
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    # Handle potential zero vectors to avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms) # Replace 0 norm with a tiny number
    normed_embeds = embeddings_np / norms
    # Calculate the dot product matrix (cosine similarity for normalized vectors)
    similarity_matrix = np.dot(normed_embeds, normed_embeds.T)
    logger.debug("Similarity matrix calculated.")

    # 3. Find pairs exceeding the threshold
    redundant_issues = []
    num_chunks = similarity_matrix.shape[0]
    # Use a set to keep track of chunks already flagged as part of a redundant pair to avoid duplicate reporting
    # E.g., if (1, 5) is flagged, don't also flag (5, 1) or (1, 7) if 7 is also similar.
    # This approach flags the first instance of high similarity found for a chunk.
    flagged_indices = set()

    # Iterate through the upper triangle of the similarity matrix (excluding the diagonal)
    for i in range(num_chunks):
        # Skip if this chunk has already been flagged as part of a redundant pair
        if i in flagged_indices:
            continue
        for j in range(i + 1, num_chunks):
            # Skip if the second chunk has already been flagged
            if j in flagged_indices:
                continue

            similarity = similarity_matrix[i, j]
            # Check if similarity exceeds the threshold
            if similarity > similarity_threshold:
                # Get snippets from both chunks for context
                snippet_i = chunks[i][:100].replace('\n', ' ') # First 100 chars, newlines replaced
                snippet_j = chunks[j][:100].replace('\n', ' ')

                # Create the issue dictionary
                issue = {
                    "Issue Type": "Potentially Redundant Content",
                    "Details": f"Chunk {i} and Chunk {j} are highly similar (Similarity: {similarity:.3f})",
                    "Excerpt": f"Chunk {i}: '{snippet_i}...'\n---\nChunk {j}: '{snippet_j}...'"
                }
                redundant_issues.append(issue)

                # Mark both chunks as flagged so they aren't reported again in other pairs
                flagged_indices.add(i)
                flagged_indices.add(j)

                # Optimization: Once chunk 'i' is found to be redundant with 'j',
                # break the inner loop and move to the next 'i'.
                # This prevents reporting multiple redundancies for the same chunk 'i'.
                break

    logger.info(f"Found {len(redundant_issues)} potential redundancy issues (pairs of similar chunks).")
    return redundant_issues

def analyze_document(file_path):
    """
    Performs a full analysis on a given PDF document, checking for outdated references
    and redundant content. Saves the findings to an Excel file.

    Args:
        file_path (str): The path to the PDF file to be analyzed.

    Returns:
        str | None: The path to the generated Excel report if issues were found and saved,
                    otherwise None.
    """
    logger.info(f"Starting document analysis for: {file_path}")

    # Ensure embedder is ready (needed for redundancy check)
    analyzer_embedder = initialize_embedder()
    if analyzer_embedder is None:
        logger.error("Analysis aborted: Embedder could not be initialized.")
        return None

    # 1. Extract text from the PDF
    text = extract_text_from_pdf(file_path)
    if text is None:
        logger.error(f"Analysis aborted: Could not extract text from {file_path}.")
        return None
    if not text.strip():
        logger.warning(f"Analysis warning: Extracted text from {file_path} is empty.")
        # Decide if analysis should proceed or stop
        # return None # Option to stop if text is empty

    # 2. Perform outdated reference check
    outdated = find_outdated_references(text)

    # 3. Perform redundancy check (requires splitting into chunks first)
    chunks = split_text_into_chunks(text)
    redundant = [] # Initialize as empty list
    if chunks: # Only run redundancy check if chunks were created
        redundant = find_redundant_content(chunks, analyzer_embedder)
    else:
        logger.warning("No chunks generated, skipping redundancy check.")

    # 4. Combine and report results
    all_issues = outdated + redundant

    if not all_issues:
        logger.info("Analysis complete. No potentially outdated or redundant content found based on criteria.")
        return None
    else:
        logger.info(f"Analysis complete. Found {len(all_issues)} potential issues.")
        # Create a Pandas DataFrame from the list of issues
        try:
            df = pd.DataFrame(all_issues)
            # Define the output file path using config
            output_filename = config.ANALYSIS_OUTPUT_FILE
            logger.info(f"Attempting to export analysis results to Excel file: {output_filename}")
            # Export the DataFrame to an Excel file, without the DataFrame index
            df.to_excel(output_filename, index=False)
            logger.info(f"Analysis results successfully exported to '{output_filename}'")
            return output_filename # Return the path to the generated report
        except ImportError:
             logger.error("Analysis export failed: `pandas` or `openpyxl` library not found. Please install them (`pip install pandas openpyxl`).")
             return None
        except Exception as e:
            logger.error(f"Error exporting analysis results to Excel: {e}", exc_info=True)
            return None

# --- Allow running this module as a standalone script ---
if __name__ == "__main__":
    import sys
    # Configure basic logging when run as a script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if a file path argument is provided
    if len(sys.argv) > 1:
        pdf_path_arg = sys.argv[1]
        if os.path.exists(pdf_path_arg):
            # Run the analysis on the provided PDF file
            analyze_document(pdf_path_arg)
        else:
            print(f"Error: File not found at the specified path: {pdf_path_arg}")
            logger.error(f"File not found at path provided via command line: {pdf_path_arg}")
    else:
        # Print usage instructions if no file path is given
        print("\nUsage: python -m app.document_analyzer <path_to_pdf_file>")
        print("Example: python -m app.document_analyzer \"./documents/my_policy.pdf\"\n")

