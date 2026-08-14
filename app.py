import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="TeleRAG",
    page_icon="📡"
)

st.title("📡 TeleRAG")
st.write("3GPP Telecom Standards Assistant")


@st.cache_resource
def load_components():

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.load_local(
        "vector_store",
        embedding,
        allow_dangerous_deserialization=True
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    return vectorstore, llm


vectorstore, llm = load_components()

query = st.text_input("Ask a Telecom Question")


if st.button("Submit") and query:

    with st.spinner("Searching documents..."):

        results = vectorstore.similarity_search(
            query,
            k=20
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
You are a Telecom 3GPP assistant.

Answer ONLY from the provided context.
Do not use external knowledge.

If the answer is not present in the context, reply exactly:
Information not available in provided documents.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    with st.expander("Sources"):

        for i, doc in enumerate(results, start=1):

            page = doc.metadata.get("page", "N/A")

            st.markdown(f"**Source {i} (Page {page})**")

            st.write(doc.page_content[:500])

            st.divider()