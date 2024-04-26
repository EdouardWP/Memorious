from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import gradio as gr
from gradio_pdf import PDF
import os

api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the MISTRAL_API_KEY environment variable.")

llm = MistralAI(api_key=api_key, model="open-mistral-7b")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model

def qa(question: str, doc) -> str:
    if not doc:
        return "No document uploaded."
    file_path = doc.name  # Get the temporary file path

    try:
        my_pdf = SimpleDirectoryReader(input_files=[file_path]).load_data()
        my_pdf_index = VectorStoreIndex.from_documents(my_pdf)
        my_pdf_engine = my_pdf_index.as_query_engine()
        response = my_pdf_engine.query(question)
        if not response:
            return "No answer found in the document."
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

demo = gr.Interface(
    fn=qa,
    inputs=[
        gr.Text(label="Question", placeholder="Type your question here"), 
        gr.File(label="Upload Document")
    ],
    outputs=gr.Text(label="Memor's Answer", lines=15),
    title="Memorious Tech Demo",
    description="Upload a PDF document and ask a question about its content now, or later.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
