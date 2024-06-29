from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import PyPDF2
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)
load_dotenv()
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# Function to scrape website for new links
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Assuming the links are in <a> tags
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# Function to fetch and save content of a link as a text file
def fetch_and_save_content(link, output_path):
    response = requests.get(link)
    with open(output_path, 'w') as text_file:
        text_file.write(response.text)

# Function to convert text file content to embeddings and update FAISS
def process_new_text_file(txt_path):
    with open(txt_path, 'r') as file:
        file_text = file.read()
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64
    )
    texts = text_splitter.split_text(file_text)
    new_documents = [
        Document(page_content=chunked_text, metadata={"doc_title": txt_path, "chunk_num": i})
        for i, chunked_text in enumerate(texts)
    ]
    vector_store.add_documents(new_documents)

# Ingest initial data function
def ingest_data():
    files = os.listdir("../../Aihacks/txt_files")
    file_texts = []
    for file in files:
        with open(f"../../Aihacks/txt_files/{file}") as f:
            file_text = f.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=64
        )
        texts = text_splitter.split_text(file_text)
        for i, chunked_text in enumerate(texts):
            file_texts.append(Document(page_content=chunked_text, 
                    metadata={"doc_title": file.split(".")[0], "chunk_num": i}))

    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(
        file_texts,
        embedding=embeddings
    )
    return vector_store

# Initialize FAISS vector store
vector_store = ingest_data()
retriever = vector_store.as_retriever()
llm = OctoAIEndpoint(
    model="meta-llama-3-8b-instruct",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)

template = """You are an investment banker please give advice. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def index():
    return "Hello, this is the backend for the AI-powered Q&A."

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    answer = chain.invoke(question)
    return jsonify({'answer': answer})

def check_for_new_links():
    urls_to_monitor = [
        'http://127.0.0.1:8080',  # Replace with actual URLs
    ]
    for url in urls_to_monitor:
        new_links = scrape_website(url)
        for link in new_links:
            # Create a unique text file name based on the link
            txt_path = f"../../Aihacks/txt_files/{link.split('/')[-1]}.txt"
            fetch_and_save_content(link, txt_path)
            process_new_text_file(txt_path)

scheduler = BackgroundScheduler()
scheduler.add_job(func=check_for_new_links, trigger="interval", minutes=10)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True, port=8080)