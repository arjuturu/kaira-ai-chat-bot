from app.services.document_service import extract_text
from app.services.vector_store import build_vector_store


def handle_file_upload(file):
    if file is None:
        return "Please upload a file.", None, None, ""

    document_text = extract_text(file)
    vector_store = build_vector_store(document_text)

    # Clear summary on new upload
    return "✅ File loaded successfully.", vector_store, document_text, ""
