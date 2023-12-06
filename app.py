# We are going to develop the code for the RAG here. This is going to be the first and the only attempt IA!!
# To create the POC

# ! We need to do the following,
# Convert the PDF to Embeddings and save it into a vector database.
# Load LLAMA 2
# Connect LLAMA 2 to the vector database.
# Ask Questions and give answers.


# ! LLAMA IS LOADED
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import json

from torch import cuda
import torch
import transformers
from time import time
import chromadb
from chromadb.config import Settings
from langchain.llms import huggingface_pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Change the model path here to test any other model.
# model_path = 'training_date_02_10_2023_psql/final_merged_checkpoint'
model_path = 'Llama-13b-chat'

tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder="offload/",
                cache_dir="cache/",
                quantization_config=nf4_config # forgot this on the first try so full model was loaded.
)

model_config = transformers.AutoConfig.from_pretrained(model_path)


# define query huggingface_pipeline
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm = huggingface_pipeline.HuggingFacePipeline(pipeline=query_pipeline)


# Ingestion of data using text loader
loader = TextLoader("MTM_Memoir_txt.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# let's create the embeddings and store in vector store
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# initialize chromadb
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# Initialize the chain
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

# let's test the RAG
def test_rag(qa, query):
    print(query)
    result = qa.run(query)
    print(f"Result \t {result}")

test_rag(qa, "Hello when were you born?")


def preprocess_query(query):

    # load the query as a dict
    # res = json.loads(str(query))

    # human_language = res['human_language']

    # SQL_TABLE_CONTEXT = "CREATE TABLE properties (address character, details characterstate character, property_type character, price integer, bedrooms integer, bathrooms integer, sqft integer)"

    # INTRO = f"<s>[INST] <<SYS>> \
    #         You are a helpful, genius data scientist, who has access to a database that contains listing of properties in New York. Your job is to write PostgreSQL Query to fetch data based on User Request and Parameters. If you can't reply correctly just say that not enough information was provided \n\n<</SYS>>"
    INTRO = f"<s>[INST] <<SYS>>You are former Malaysian Prime Minister Tun Dr Mahathir Mohamad.. A visionary leader \n\n<</SYS>>"
    INSTRUCTION = f"### Instruction\n Respond to the following query by your subject {query} Just like yourself. \n\n"
    
    RESPONSE = f"### Response:\n\n"

    final_payload = INTRO + INSTRUCTION + RESPONSE
    payload_length = len(final_payload)

    return final_payload


def get_result(qa=qa, query = ""):
    return qa.run(query)


def predict(query):
    processed_query = preprocess_query(query=query)
    result = get_result(query=processed_query)
    return(result)


# ! The following will also work now! I mistakenly wrote ap_name insted of api_name in the submit_button.click()

with gr.Blocks() as sql_generator:
    query = gr.Textbox(label="Query", placeholder='Ask the president?')

    output = gr.Textbox(label="Output")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=predict,
                        inputs=query,
                        outputs=output, api_name="predict"
                        )



sql_generator.launch()