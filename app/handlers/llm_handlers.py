from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


def handle_llm_chat(message, history, user_api_key):

    if not user_api_key:
        return "⚠️ Please enter your OpenAI API key."

    try:
        llm = ChatOpenAI(
            temperature=0,
            api_key=user_api_key
        )

        messages = []

        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        response = llm.invoke(messages)

        return response.content

    except Exception as e:
        error_text = str(e).lower()

        if "401" in error_text:
            return "❌ Invalid API key."

        if "rate limit" in error_text:
            return "⚠️ Rate limit exceeded."

        return "⚠️ Unable to process request."
