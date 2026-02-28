from app.services.chat_service import rag_chat


def handle_rag_chat(message, history, vector_store):
    # Check if user typed anything
    if not message or not message.strip():
        return "⚠️ Please type a question before submitting."

    if vector_store is None:
        return "Please upload a document to begin."
    print(f"Received message: {message}")
    return str(rag_chat(message, vector_store))