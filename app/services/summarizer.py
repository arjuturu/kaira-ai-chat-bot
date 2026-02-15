from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

def summarize(text):
    prompt = f"Summarize the following document:\n\n{text[:6000]}"
    response = llm.invoke(prompt)
    return response.content
