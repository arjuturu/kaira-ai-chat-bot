import gradio as gr

from app.handlers.upload_handlers import safe_handle_file_upload
from app.handlers.rag_handlers import handle_rag_chat
from app.handlers.llm_handlers import handle_llm_chat
from app.services.summarizer import summarize


def launch_app():
    with gr.Blocks() as demo:

        vector_state = gr.State(None)
        document_state = gr.State(None)

        # Top header
        gr.Markdown(
            """
# 🤖 **Kaira AI Chat Bot**

Welcome to Kaira AI, your intelligent assistant for document interaction and general conversations!
            """
        )

        # Mode selector
        mode_selector = gr.Radio(
            choices=[("📄 Document Chat", "rag"), ("🤖 LLM Chat", "llm")],
            value="rag",
            label="Select chat mode"
        )

        # Document Chat section
        with gr.Column(visible=True) as rag_section:
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

            gr.ChatInterface(
                fn=handle_rag_chat,
                additional_inputs=[vector_state],
                chatbot=gr.Chatbot(height=280),
            )

        # LLM Chat section
        with gr.Column(visible=False) as llm_section:
            gr.Markdown("⚠️ Enter your OpenAI API key to continue.")
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password"
            )
            gr.Markdown(
                "🔑 [Get your OpenAI API key here](https://platform.openai.com/account/api-keys)"
            )

            gr.ChatInterface(
                fn=handle_llm_chat,
                additional_inputs=[api_key_input],
                chatbot=gr.Chatbot(height=280),
            )

        # Mode switch logic
        def switch_mode(mode):
            if mode == "rag":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        mode_selector.change(
            switch_mode,
            inputs=mode_selector,
            outputs=[rag_section, llm_section]
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    launch_app()