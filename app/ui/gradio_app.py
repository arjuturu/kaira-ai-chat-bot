import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.services.document_service import extract_text
from app.services.vector_store import build_vector_store
from app.services.summarizer import summarize
from app.services.chat_service import rag_chat

vector_state = gr.State(None)
document_state = gr.State(None)



# -----------------------------
# FILE UPLOAD HANDLER (RAG TAB)
# -----------------------------
def on_file_upload(file):
    global vector_store, document_text

    if file is None:
        return "Please upload a file."

    document_text = extract_text(file)
    vector_store = build_vector_store(document_text)

    return "✅ File loaded and ready for questions to be answered.", vector_store, document_text


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
def rag_chat_handler(message, history, vector_store):
    if vector_store is None:
        return "Please upload a document(.txt, .docx and .pdf)."
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
        title="AI Document Chatbot") as demo:

        with gr.Column(elem_classes="container"):

            gr.Markdown(
                "## Kaira AI Chatbot\n"
                "Ask/Summarize your documents. Or bring your own key to ask LLM directly."
            )

            # Mode Selector
            mode_selector = gr.Radio(
                ["RAG Mode", "LLM Mode"],
                value="📄 RAG Mode",
                label="Choose Mode",
                scale=1
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

                file_upload.change( on_file_upload, inputs=file_upload, outputs=[status, vector_state, document_state] )
                summary_btn.click(on_summary, None, summary_output)

                gr.Markdown("### Ask Your Document")

                gr.ChatInterface(
                    fn=rag_chat_handler,
                    chatbot=gr.Chatbot(height=320),
                    additional_inputs=[vector_state]
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
    server_name="0.0.0.0",
    server_port=7860,
    css="""
    body {
        font-size: 16px;
    }

    .gradio-container {
        max-width: 100% !important;
        padding: 12px !important;
    }

    button {
        font-size: 16px !important;
        padding: 10px 14px !important;
    }

    textarea {
        font-size: 16px !important;
    }

    input {
        font-size: 16px !important;
    }

    .chatbot {
        height: 420px !important;
    }
    """
)


