## This code is for getting the file name from which the provided query belongs. The query contains a context from any one of the documents
## and the code should output that file

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the name of the file in which the provided question/context belongs.
    Also provide the whole context about the provided context.
    <context>
    {context}
    <context>
    Question:{input}
    """
)


def create_vector_embedding():
    if "vectors" not in st.session_state: ## session_state helps to remember vectorStore DB

        st.session_state.embeddings=NVIDIAEmbeddings() ## Embedding

        st.session_state.loader=PyPDFDirectoryLoader("papers") ## Data Ingestion step

        st.session_state.docs=st.session_state.loader.load() ## Document Loading

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) 

        # Attach file name to each document's metadata
        for doc in st.session_state.final_documents:
            file_name = doc.metadata.get('source')  # Assuming 'source' contains the file name
            doc.metadata["file_name"] = file_name

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("File Name Extractor")

user_prompt = st.text_input("Enter your query from any of the papers")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt) ## Creates a chain for passing a list of documents to a model.
    ## it sends all the list of documents to the {context} (under prompt)
    
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':user_prompt})

    # st.write(response['answer'])
    
    # st.write(response)
    
    if "context" in response and isinstance(response["context"], list):
        try:
            # Extract documents from the context
            documents = response["context"]
            
            # Ensure that each item in documents is of type Document
            if all(isinstance(doc, Document) for doc in documents):
                # Extract metadata and content from the first document
                first_doc = documents[0]
                metadata = first_doc.metadata
                file_name = metadata.get("file_name", "Unknown file")
                page_content = first_doc.page_content

                st.write(f"File Name: {file_name}")
                st.write(f"Context: {page_content}")
            else:
                st.write("The context does not contain valid Document objects.")
        except Exception as e:
            st.write(f"Error processing response: {e}")
    else:
        st.write("No valid context found in the response.")
    