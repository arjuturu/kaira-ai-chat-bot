import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.services.document_service import extract_text
from app.services.vector_store import build_vector_store
from app.services.summarizer import summarize
from app.services.chat_service import rag_chat

vector_store = None
document_text = None


# -----------------------------
# FILE UPLOAD HANDLER (RAG TAB)
# -----------------------------
def on_file_upload(file):
    global vector_store, document_text

    if file is None:
        return "Please upload a file."

    document_text = extract_text(file)
    vector_store = build_vector_store(document_text)

    return "✅ File loaded and ready for questions to be answered."


# -----------------------------
# SUMMARY HANDLER
# -----------------------------
def on_summary():
    global document_text

    if not document_text:
        return "No document loaded."

    return summarize(document_text)


# -----------------------------
# RAG CHAT HANDLER
# -----------------------------
def rag_chat_handler(message, history):
    global vector_store
    return rag_chat(message, vector_store)


# -----------------------------
# LLM CHAT HANDLER (BYO KEY)
# -----------------------------
def llm_chat_handler(message, history, user_api_key):
    if not user_api_key:
        return "⚠️ Please enter your OpenAI API key. You can get one from https://platform.openai.com/settings/organization/api-keys"

    try:
        llm = ChatOpenAI(
            temperature=0,
            api_key=user_api_key
        )

        messages = []

        # Convert Gradio history to LangChain format
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

        if "401" in error_text or "invalid_api_key" in error_text:
            return "❌ Invalid API key. Please check your key and try again."

        if "rate limit" in error_text:
            return "⚠️ Rate limit exceeded. Please try again later."

        return "⚠️ Unable to process request. Please verify your API key."


# -----------------------------
# MAIN APP
# -----------------------------
def launch_app():
    with gr.Blocks(
        title="AI Document Chatbot",) as demo:

        with gr.Column(elem_classes="container"):

            gr.Markdown(
                "## 📘 Kaira AI Chatbot\n"
                "Ask/Summarize your documents. Or bring your own key to ask LLM directly."
            )

            # Mode Selector
            mode_selector = gr.Radio(
                ["📄 RAG Mode", "🤖 LLM Mode"],
                value="📄 RAG Mode",
                label="Choose Mode"
            )

            # ---------------- RAG SECTION ----------------
            with gr.Column(visible=True) as rag_section:

                file_upload = gr.File(
                    label="Upload Document (PDF / DOCX / TXT)"
                )

                status = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                summary_btn = gr.Button("Generate Summary")
                summary_output = gr.Textbox(
                    lines=5,
                    label="Summary"
                )

                file_upload.change(on_file_upload, file_upload, status)
                summary_btn.click(on_summary, None, summary_output)

                gr.Markdown("### Ask Your Document")

                gr.ChatInterface(
                    fn=rag_chat_handler,
                    chatbot=gr.Chatbot(height=400),
                )

            # ---------------- LLM SECTION ----------------
            with gr.Column(visible=False) as llm_section:

                gr.Markdown(
                    "⚠️ Bring your own OpenAI API key. You can get one from https://platform.openai.com/settings/organization/api-keys"
                )

                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password"
                )

                gr.ChatInterface(
                    fn=llm_chat_handler,
                    additional_inputs=[api_key_input],
                    chatbot=gr.Chatbot(height=400),
                )

            # Toggle Logic
            def toggle_sections(selected):
                if "RAG" in selected:
                    return gr.update(visible=True), gr.update(visible=False)
                return gr.update(visible=False), gr.update(visible=True)

            mode_selector.change(
                toggle_sections,
                mode_selector,
                [rag_section, llm_section]
            )

    # demo.launch(share=True)
    demo.launch(
    css="""
    body { background-color: #f9fafb; }
    .container { max-width: 700px; margin: auto; padding: 20px; }
    .mode-buttons { display: flex; gap: 10px; }
    .mode-buttons button { flex: 1; }
    .error-border textarea { border: 2px solid red !important; }
    """,
    share=True
)

