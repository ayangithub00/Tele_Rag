import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# ... baaki code
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = FAISS.load_local(
    "vector_store",
    embedding,
    allow_dangerous_deserialization=True
)

print("FAISS loaded successfully")

results = vectorstore.similarity_search(
    "What is AMF?",
    k=5
)

print(f"Retrieved {len(results)} documents")

for i, doc in enumerate(results, 1):
    print(f"\nSource {i}")
    print(doc.page_content[:300])

context = "\n\n".join(
    [doc.page_content for doc in results]
)

print("\nContext created successfully")
print(f"Context length: {len(context)}")