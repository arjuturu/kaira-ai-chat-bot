import gradio as gr
from datetime import datetime

from app.handlers.upload_handlers import safe_handle_file_upload
from app.handlers.rag_handlers import handle_rag_chat
from app.handlers.llm_handlers import handle_llm_chat
from app.services.summarizer import summarize


def timestamped(role, message):
    now = datetime.now().strftime("%d %b %H:%M")
    return {"role": role, "content": f"[{now}] {message}"}


def launch_app():
    with gr.Blocks() as demo:

        # States for persistence
        vector_state = gr.State(None)
        document_state = gr.State(None)
        rag_history = gr.State([])
        llm_history = gr.State([])

        # Header
        gr.Markdown(
            """
# 🚀 **Kaira AI Chat Bot**

Welcome to Kaira AI, your intelligent assistant for document interaction and general conversations!
            """
        )

        # Document Chat Tab
        with gr.Tab("📄 Document Chat"):
            file_upload = gr.File(
                label="Upload (.pdf, .docx, .txt)",
                file_types=[".pdf", ".docx", ".txt"]
            )
            status = gr.Markdown("")
            summary_btn = gr.Button("Summarize")
            summary_output = gr.Textbox(lines=4, interactive=False)

            file_upload.change(
                safe_handle_file_upload,
                inputs=file_upload,
                outputs=[status, vector_state, document_state, summary_output]
            )

            summary_btn.click(summarize, inputs=document_state, outputs=summary_output)

            def rag_chat(message, history, vector_state, rag_history):
                reply = handle_rag_chat(message, history, vector_state)
                rag_history.append(timestamped("user", message))
                rag_history.append(timestamped("assistant", reply))
                return rag_history

            gr.ChatInterface(
                fn=lambda msg, hist: rag_chat(msg, hist, vector_state.value, rag_history.value),
                chatbot=gr.Chatbot(height=280),
                textbox=gr.Textbox(
                    label="Ask about your document",
                    placeholder="🔍 Ask me to summarize, explain, or extract insights..."
                )
            )

        # LLM Chat Tab
        with gr.Tab("🧠 LLM Chat"):
            gr.Markdown("⚠️ Enter your OpenAI API key to continue.")
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            gr.Markdown("🔑 [Get your OpenAI API key here](https://platform.openai.com/account/api-keys)")

            def llm_chat(message, history, api_key, llm_history):
                reply = handle_llm_chat(message, history, api_key)
                llm_history.append(timestamped("user", message))
                llm_history.append(timestamped("assistant", reply))
                return llm_history

            gr.ChatInterface(
                fn=lambda msg, hist: llm_chat(msg, hist, api_key_input.value, llm_history.value),
                chatbot=gr.Chatbot(height=280),
                textbox=gr.Textbox(
                    label="Ask the language model",
                    placeholder="🧠 Ask me anything — coding help, brainstorming, or casual chat..."
                )
            )

        # Footer branding
        gr.Markdown(
            """
---
**Built by Amaranathareddy Juturu** | Built with ❤️ using OpenAI, LangChain, Faiss, and Gradio.  
[GitHub Repo](https://github.com/arjuturu/kaira-ai-chat-bot)
            """
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    launch_app()