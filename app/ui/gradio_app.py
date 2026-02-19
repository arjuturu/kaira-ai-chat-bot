import gradio as gr

from app.handlers.upload_handlers import handle_file_upload
from app.handlers.rag_handlers import handle_rag_chat
from app.handlers.llm_handlers import handle_llm_chat
from app.services.summarizer import summarize


def launch_app():
    with gr.Blocks(title="Kaira AI") as demo:

        vector_state = gr.State(None)
        document_state = gr.State(None)

        gr.Markdown("## 🤖 Kaira AI")

        with gr.Row():
            rag_btn = gr.Button("📄 Document Chat", variant="primary")
            llm_btn = gr.Button("🤖 LLM Chat")

        with gr.Column(visible=True) as rag_section:

            file_upload = gr.File(label="Upload (.pdf, .docx, .txt)")
            status = gr.Markdown("")
            summary_btn = gr.Button("Summarize")
            summary_output = gr.Textbox(lines=4, interactive=False)

            file_upload.change(
                handle_file_upload,
                inputs=file_upload,
                outputs=[status, vector_state, document_state,summary_output]
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

        with gr.Column(visible=False) as llm_section:

            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password"
            )

            gr.ChatInterface(
                fn=handle_llm_chat,
                additional_inputs=[api_key_input],
                chatbot=gr.Chatbot(height=280),
            )

        def show_rag():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(variant="primary"),
                gr.update(variant="secondary")
            )

        def show_llm():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(variant="secondary"),
                gr.update(variant="primary")
            )

        rag_btn.click(
            show_rag,
            None,
            [rag_section, llm_section, rag_btn, llm_btn]
        )

        llm_btn.click(
            show_llm,
            None,
            [rag_section, llm_section, rag_btn, llm_btn]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
