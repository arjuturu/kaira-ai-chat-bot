from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(temperature=0)


def llm_chat(message: str, user_api_key: str):
    llm = ChatOpenAI(temperature=0,api_key=user_api_key)
    response = llm.invoke(message)
    return response.content


def rag_chat(message: str, vector_store):
    if vector_store is None:
        return "No document uploaded yet."

    docs = vector_store.similarity_search(message, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content="Answer ONLY from the provided context."),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {message}"
        ),
    ]

    response = llm.invoke(messages)

    return response.content
