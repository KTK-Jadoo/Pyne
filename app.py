import gradio as gr
from confident_learning_app.app import app as build_app


# For Hugging Face Spaces, expose a global `demo` Blocks instance
demo: gr.Blocks = build_app()


if __name__ == "__main__":
    demo.launch()

