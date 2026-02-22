import gradio as gr
from datetime import datetime

from app.handlers.upload_handlers import safe_handle_file_upload
from app.handlers.rag_handlers import handle_rag_chat
from app.handlers.llm_handlers import handle_llm_chat
from app.services.summarizer import summarize


def timestamped(role, message):
    """Helper to format messages with timestamp."""
    now = datetime.now().strftime("%d %b %H:%M")
    return {"role": role, "content": f"[{now}] {message}"}


def launch_app():
    with gr.Blocks() as demo:

        vector_state = gr.State(None)
        document_state = gr.State(None)

        # Header
        gr.Markdown(
            """
# 🤖 **Kaira AI Chat Bot**

Welcome to Kaira AI, your intelligent assistant for document interaction and general conversations!
            """
        )

        # Mode selector (default = Document Chat)
        mode_selector = gr.Radio(
            choices=[("📄 Document Chat", "rag"), ("🤖 LLM Chat", "llm")],
            value="rag",
            label="Select chat mode"
        )

        # Document Chat section
        rag_section = gr.Group(visible=True)
        with rag_section:
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

            summary_btn.click(
                summarize,
                inputs=document_state,
                outputs=summary_output
            )

            rag_chatbot = gr.Chatbot(height=280)
            rag_input = gr.Textbox(
                label="Ask about your document",
                placeholder="🔍 Ask me to summarize, explain, or extract insights..."
            )
            rag_send = gr.Button("Send")
            rag_clear = gr.Button("Clear Chat")

            def rag_chat(message, history, vector_state):
                reply = handle_rag_chat(message, history, vector_state)
                history.append(timestamped("user", message))
                history.append(timestamped("assistant", reply))
                return history

            def clear_rag_chat():
                return []

            rag_send.click(rag_chat, inputs=[rag_input, rag_chatbot, vector_state], outputs=rag_chatbot)
            rag_clear.click(clear_rag_chat, outputs=rag_chatbot)

        # LLM Chat section
        llm_section = gr.Group(visible=False)
        with llm_section:
            gr.Markdown("⚠️ Enter your OpenAI API key to continue.")
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            gr.Markdown("🔑 [Get your OpenAI API key here](https://platform.openai.com/account/api-keys)")

            llm_chatbot = gr.Chatbot(height=280)
            llm_input = gr.Textbox(
                label="Ask the language model",
                placeholder="🤖 Ask me anything — coding help, brainstorming, or casual chat..."
            )
            llm_send = gr.Button("Send")
            llm_clear = gr.Button("Clear Chat")

            def llm_chat(message, history, api_key):
                reply = handle_llm_chat(message, history, api_key)
                history.append(timestamped("user", message))
                history.append(timestamped("assistant", reply))
                return history

            def clear_llm_chat():
                return []

            llm_send.click(llm_chat, inputs=[llm_input, llm_chatbot, api_key_input], outputs=llm_chatbot)
            llm_clear.click(clear_llm_chat, outputs=llm_chatbot)

        # Corrected mode switch logic
        def switch_mode(mode):
            if mode == "rag":
                return (
                    gr.update(visible=True),   # rag_section
                    gr.update(visible=False),  # llm_section
                    gr.update(visible=True),   # rag_chatbot
                    gr.update(visible=False),  # llm_chatbot
                    gr.update(visible=True),   # rag_input
                    gr.update(visible=False),  # llm_input
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )

        mode_selector.change(
            switch_mode,
            inputs=mode_selector,
            outputs=[rag_section, llm_section, rag_chatbot, llm_chatbot, rag_input, llm_input]
        )

        # Persistent footer branding
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