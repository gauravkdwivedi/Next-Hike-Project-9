import os
import pickle
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set the OpenAI API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

def process_urls(urls):
    """
    Load data from the provided URLs, split the data into chunks,
    create embeddings, and store them in a FAISS index.

    Args:
        urls (list): List of URLs to process.

    Returns:
        FAISS: The FAISS vectorstore containing the document embeddings.
    """    
    # Load data from the URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    # Split data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    
    # Create embeddings from the document chunks
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    
    return vectorstore_openai

def save_vectorstore(vectorstore, file_path):
    """
    Save the FAISS vectorstore to a file using pickle.

    Args:
        vectorstore (FAISS): The FAISS vectorstore to save.
        file_path (str): The file path where the vectorstore will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump((vectorstore.docstore, vectorstore.index_to_docstore_id, vectorstore.index), f)

def load_vectorstore(file_path):
    """
    Load the FAISS vectorstore from a file using pickle.

    Args:
        file_path (str): The file path from where the vectorstore will be loaded.

    Returns:
        FAISS: The loaded FAISS vectorstore.
    """
    with open(file_path, 'rb') as f:
        docstore, index_to_docstore_id, index = pickle.load(f)
    
    # Initialize the FAISS vectorstore with the loaded components and embedding function
    vectorstore = FAISS(docstore=docstore, index_to_docstore_id=index_to_docstore_id, index=index, embedding_function=OpenAIEmbeddings())
    return vectorstore

def create_qa_chain(vectorstore):
    """
    Create a QA (Question Answering) chain with sources using the FAISS vectorstore.

    Args:
        vectorstore (FAISS): The FAISS vectorstore to use for the retriever.

    Returns:
        RetrievalQAWithSourcesChain: The QA chain object.
    """
    # Create the retrieval QA chain with the LLM and retriever from the vectorstore
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return chain