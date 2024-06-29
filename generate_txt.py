from urllib.request import urlopen
import html2text
url='https://www.microsoft.com/en-us/investor/earnings/fy-2024-q3/press-release-webcast'
page = urlopen(url)
html_content = ""
if page.headers.get_content_charset():
    html_content = page.read().decode(page.headers.get_content_charset())
else:
    html_content = page.read().decode("utf-8")
rendered_content = html2text.html2text(html_content)
file = open('./txt_files/file_text.txt', 'w')
file.write(rendered_content)
file.close()

"""
# Ingest Data
"""

# %%
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# %%
files = os.listdir("../Aihacks/txt_files")
file_texts = []
for file in files:
    with open(f"../Aihacks/txt_files/{file}") as f:
        file_text = f.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64, 
    )
    texts = text_splitter.split_text(file_text)
    for i, chunked_text in enumerate(texts):
        file_texts.append(Document(page_content=chunked_text, 
                metadata={"doc_title": file.split(".")[0], "chunk_num": i}))

# %%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# %%
embeddings = HuggingFaceEmbeddings()

# %%
vector_store = FAISS.from_documents(
    file_texts,
    embedding=embeddings
)