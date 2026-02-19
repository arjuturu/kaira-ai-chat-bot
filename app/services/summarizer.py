from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

def summarize(text: str):
    if not text:
        return "No text provided to summarize."
    prompt = f"Summarize the following document:\n\n{text[:6000]}"
    response = llm.invoke(prompt)
    return response.content
