# %%
"""
# Setup
"""

# %%
# OctoAI
# ! pip install langchain langchain-community faiss-cpu sentence-transformers octoai-sdk langchain-text-splitters lxml tiktoken python-dotenv 'arize-phoenix[evals]'

# %%
from dotenv import load_dotenv
import os

load_dotenv()
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# %%
"""
# Ingest Data
"""

# %%
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# %%
files = os.listdir("../city_data")
file_texts = []
for file in files:
    with open(f"../city_data/{file}") as f:
        file_text = f.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64, 
    )
    texts = text_splitter.split_text(file_text)
    for i, chunked_text in enumerate(texts):
        file_texts.append(Document(page_content=chunked_text, 
                metadata={"doc_title": file.split(".")[0], "chunk_num": i}))

# %%
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# %%
embeddings = HuggingFaceEmbeddings()

# %%
vector_store = FAISS.from_documents(
    file_texts,
    embedding=embeddings
)

# %%
"""
# Search the Data
"""

# %%
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
llm = OctoAIEndpoint(
        model="meta-llama-3-8b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
    )

# %%
retriever = vector_store.as_retriever()

# %%
from langchain.prompts import ChatPromptTemplate
template="""You are a tour guide. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# %%
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
chain.invoke("What is the worst metro line in Paris?")

# %%
