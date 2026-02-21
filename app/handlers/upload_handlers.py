from app.services.document_service import extract_text
from app.services.vector_store import build_vector_store
import os

def safe_handle_file_upload(file):
    if file is None:
        return "Please upload a file to begin.", None, None, ""
    
    max_size_mb = 2  # set your limit here
    file_size_mb = os.path.getsize(file.name) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        return (
            f"⚠️ File too large! Limit is {max_size_mb} MB.",
            None,  # vector_state
            None,  # document_state
            ""     # summary_output
        )

    # Call your original handler if file size is acceptable
    return handle_file_upload(file)

def handle_file_upload(file):
    if file is None:
        return "Please upload a file to begin.", None, None, ""

    document_text = extract_text(file)
    vector_store = build_vector_store(document_text)

    # Clear summary on new upload
    return "✅ File loaded successfully.", vector_store, document_text, ""
