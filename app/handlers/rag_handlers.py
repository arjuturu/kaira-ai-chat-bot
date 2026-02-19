from app.services.chat_service import rag_chat


def handle_rag_chat(message, history, vector_store):
    if vector_store is None:
        return "Please upload a document first."

    return str(rag_chat(message, vector_store))