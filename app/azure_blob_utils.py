# app/azure_blob_utils.py
from azure.storage.blob import BlobServiceClient
import os
import logging

# Use relative import to access config variables from within the same package
from . import config

# Setup logger for this module
logger = logging.getLogger(__name__)

def get_blob_service_client():
    """
    Initializes and returns the Azure Blob Service Client using the connection string
    from the configuration. Handles potential errors during initialization.

    Returns:
        BlobServiceClient | None: The initialized client object or None if initialization fails.
    """
    connection_string = config.AZURE_CONNECTION_STRING
    if not connection_string or "PLACEHOLDER" in connection_string:
        logger.warning("Azure Connection String is not configured correctly in config or environment variables.")
        return None
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logger.info(f"Successfully initialized Azure Blob Service Client for account: {blob_service_client.account_name}")
        return blob_service_client
    except ValueError as e:
        logger.error(f"Error initializing Azure Blob Service Client due to invalid connection string: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Azure client initialization: {e}")
        return None

def upload_to_blob(file_path, blob_name, container_name=config.AZURE_CONTAINER_NAME):
    """
    Uploads a local file to the specified Azure Blob Storage container.

    Args:
        file_path (str): The local path to the file that needs to be uploaded.
        blob_name (str): The desired name for the blob in Azure storage.
        container_name (str): The name of the target container in Azure. Defaults to config.AZURE_CONTAINER_NAME.

    Returns:
        str | None: The URL of the uploaded blob if successful, otherwise None.
    """
    # Ensure the container name is valid
    if not container_name:
        logger.error("Azure container name is not configured.")
        return None

    # Get the Azure Blob Service client
    blob_service_client = get_blob_service_client()
    if not blob_service_client:
        logger.error("Failed to get Azure Blob Service Client. Cannot upload.")
        return None

    try:
        # Get a client for the specific blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        logger.info(f"Uploading '{file_path}' to Azure container '{container_name}' as blob '{blob_name}'...")
        # Open the local file in binary read mode and upload its content
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True) # Use overwrite=True to replace if blob exists
        logger.info("Upload successful.")

        # Construct the URL for the uploaded blob
        # The account name is retrieved directly from the service client object
        account_name = blob_service_client.account_name
        # Ensure account_name was retrieved successfully
        if not account_name:
             logger.error("Could not determine Azure account name from client. Cannot generate URL.")
             return None # Or return a generic success message without URL

        file_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        logger.info(f"Blob accessible at URL: {file_url}")
        return file_url

    except FileNotFoundError:
        logger.error(f"Local file not found for upload: {file_path}")
        return None
    except Exception as e:
        # Catch any other exceptions during the upload process
        logger.error(f"Error uploading file '{file_path}' to Azure Blob Storage: {e}", exc_info=True)
        return None

# Example Usage Block (only runs if the script is executed directly)
if __name__ == "__main__":
    # This block allows testing the upload function independently.
    # Create a dummy file for testing purposes.
    logger.basicConfig(level=logging.INFO) # Configure basic logging for testing
    dummy_file_path = "test_upload_example.txt"
    dummy_blob_name = "test_blob_from_script.txt"

    try:
        with open(dummy_file_path, "w") as f:
            f.write("This is a test file created for Azure upload testing.")
        print(f"Created dummy file: {dummy_file_path}")

        # Attempt to upload the dummy file
        uploaded_url = upload_to_blob(dummy_file_path, dummy_blob_name)

        if uploaded_url:
            print(f"Test upload successful. File URL: {uploaded_url}")
        else:
            print("Test upload failed. Check logs and Azure configuration.")

    finally:
        # Clean up the dummy file regardless of success or failure
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
            print(f"Removed dummy file: {dummy_file_path}")
