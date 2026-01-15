import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv

#ENV SETUP 
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

#EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#STREAMLIT UI
st.title("📄 PDF Assistant")
st.write("Summarize PDFs or ask questions from them")

api_key = st.text_input(
    "Enter your Groq API key",
    type="password"
)

#MAIN APP
if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader("temp.pdf")
            documents.extend(loader.load())

        #TEXT SPLITTING
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)

        #VECTOR STORE
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8}
        )

        # MODE SELECTION
        mode = st.radio(
            "Choose an action",
            ["📄 Summarize PDF", "❓ Ask a Question"]
        )

        #SUMMARY MODE
        if mode == "📄 Summarize PDF":
            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert document summarizer.\n\n"
                        "Use ONLY the following context to create:\n"
                        "1. A brief overview\n"
                        "2. Key points in bullet form\n"
                        "3. Important conclusions if present\n\n"
                        "{context}"
                    ),
                    ("human", "Summarize the document.")
                ]
            )

            summary_chain = create_stuff_documents_chain(
                llm,
                summary_prompt
            )

            rag_chain = create_retrieval_chain(
                retriever,
                summary_chain
            )

            if st.button("Generate Summary"):
                with st.spinner("Summarizing PDF..."):
                    response = rag_chain.invoke(
                        {"input": "Summarize the document"}
                    )

                st.subheader("📄 Document Summary")
                st.write(response["answer"])

        #Q&A MODE
        if mode == "❓ Ask a Question":
            question = st.text_input(
                "Ask a question from the PDF"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Answer the question using ONLY the following context.\n"
                        "If the answer is not in the document, say you don't know.\n\n"
                        "{context}"
                    ),
                    ("human", "{input}")
                ]
            )

            qa_chain = create_stuff_documents_chain(
                llm,
                qa_prompt
            )

            rag_chain = create_retrieval_chain(
                retriever,
                qa_chain
            )

            if question:
                with st.spinner("Finding answer..."):
                    response = rag_chain.invoke(
                        {"input": question}
                    )

                st.subheader("❓ Answer")
                st.write(response["answer"])

else:
    st.warning("Please enter your Groq API key")
