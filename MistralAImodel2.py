import os
import gradio as gr
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the MISTRAL_API_KEY environment variable.")

# Initialize the models with the API key
llm = MistralAI(api_key=api_key, model="open-mistral-7b")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

# Configure global settings with the initialized models
Settings.llm = llm
Settings.embed_model = embed_model

def qa(question: str, doc: str) -> str:
    # Load the PDF document
    my_pdf = SimpleDirectoryReader(input_files=[doc]).load_data()
    # Create an index from the document for querying
    my_pdf_index = VectorStoreIndex.from_documents(my_pdf)
    # Transform the index into a query engine
    my_pdf_engine = my_pdf_index.as_query_engine()
    # Query the document
    response = my_pdf_engine.query(question)
    return response    



# Set up the Gradio interface with imported CSS and customizations
demo = gr.Interface(
    fn=qa,
    inputs=[gr.File(label="Upload Document"), 
            gr.Text(label="Question", placeholder="Type your question here")],
    outputs= gr.Text(label="Memor's Answer", lines=15),
    title="Memorious Tech Demo",
    description="Upload a PDF document and ask a question about its content now, or later.",
    theme = "soft"
)


if __name__ == "__main__":
    demo.launch()
