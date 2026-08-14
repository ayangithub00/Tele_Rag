from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#Doc loader 

pdf_files = [
    "knowledge_base/23501-k20.pdf",
    "knowledge_base/23502-k20.pdf",
    "knowledge_base/23503-k20.pdf",
    "knowledge_base/38300-j30.pdf"
]

docs = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

#Text splitter 

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
chunks = splitter.split_documents(docs)

# Embedding 

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

#Vector 

vector_store = FAISS.from_documents(chunks,embedding)
vector_store.save_local("vector_store")