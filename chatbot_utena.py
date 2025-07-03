# Import necessary libraries
# Standard library imports
import os
import typing

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from bs4.filter import SoupStrainer
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()  # take environment variables

token = os.getenv("OPENAI_API_KEY")
if not token:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
token = str(token)  # type: ignore
model = "gpt-4o-mini"  # or your preferred model

# Streamlit UI: let the user select the knowledge source
source = st.selectbox(
    "Choose knowledge source:",
    ("Web", "Text file", "PDF", "All")
)

# --- Web source logic ---
# Allow user to input a custom URL, fallback to Wikipedia and utenainfo.lt if needed
if source in ("Web", "All"):
    custom_url = st.text_input(
        "Enter a web page URL (leave blank for default Wikipedia page):",
        value=""
    )
    docs_web = []
    web_url = None
    urls_to_try = []
    if custom_url:
        urls_to_try.append(custom_url)
    urls_to_try.extend([
        "https://en.wikipedia.org/wiki/Utena,_Lithuania",
        "https://www.utenainfo.lt/lankytinos-vietos/utena/"
    ])
    for url in urls_to_try:
        try:
            loader = WebBaseLoader(web_paths=(url,))
            docs_web = loader.load()
            if docs_web:
                web_url = url
                break
        except Exception:
            continue
    if not docs_web:
        st.warning("Could not load any web content from the provided or fallback URLs.")
else:
    docs_web = []

# --- Text file source logic ---
# Allow user to upload a .txt file, fallback to utena.txt if none uploaded
if source in ("Text file", "All"):
    uploaded_txt = st.file_uploader("Upload a .txt file", type="txt")
    docs_txt = []
    if uploaded_txt is not None:
        with open("temp_uploaded.txt", "wb") as f:
            f.write(uploaded_txt.read())
        text_loader = TextLoader("temp_uploaded.txt")
        docs_txt = text_loader.load()
    else:
        text_loader = TextLoader("utena.txt")
        docs_txt = text_loader.load()
else:
    docs_txt = []

# --- PDF source logic ---
# Allow user to upload a PDF, fallback to Utena-region-overview.pdf if none uploaded
if source in ("PDF", "All"):
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    docs_pdf = []
    if uploaded_pdf is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())
        pdf_loader = PyPDFLoader("temp_uploaded.pdf")
        docs_pdf = pdf_loader.load()
    else:
        pdf_loader = PyPDFLoader("Utena-region-overview.pdf")
        docs_pdf = pdf_loader.load()
else:
    docs_pdf = []

# --- Combine all loaded documents from selected sources ---
all_docs = docs_web + docs_txt + docs_pdf

# --- Split documents into manageable chunks for embedding ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(all_docs)

# --- Check for empty splits and show error if no data is loaded ---
if not splits:
    st.error("No documents found in the selected source(s). Please check your data or try a different source.")
    st.stop()

# --- Create Chroma vectorstore from document splits ---
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=SecretStr(typing.cast(str, token))
    )
)
retriever = vectorstore.as_retriever()

# --- Prompt template for the chatbot ---
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """
)

# --- Helper function to format docs and extract sources ---
def format_docs(docs):
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        sources.append(source)
    content = "\n\n".join(doc.page_content for doc in docs)
    return content, sources

# --- Streamlit app title ---
st.title("Home town Utena")

# --- Main function to generate a response using RAG chain ---
def generate_response(input_text):
    llm = ChatOpenAI(temperature=0.7, api_key=SecretStr(typing.cast(str, token)), model=model)

    # Get context and sources
    context, sources = format_docs(retriever.invoke(input_text))

    rag_chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(input_text)
    st.info(answer)
    st.markdown("**Sources:**")
    for src in set(sources):
        if src.startswith("http"):
            st.markdown(f"- [{src}]({src})")
        else:
            st.markdown(f"- {src}")

# --- Streamlit input and response display ---
user_input = st.text_input("Ask a question about Utena:")
if user_input:
    generate_response(user_input)
