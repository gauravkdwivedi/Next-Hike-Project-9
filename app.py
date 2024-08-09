import os
import streamlit as st
import pickle
import time
import requests
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit_authenticator as stauth
import json
import pandas as pd
import yaml
from yaml.loader import SafeLoader

# Load environment variables
load_dotenv()

# Load configuration file
config_path = 'config.yaml'
if not os.path.exists(config_path):
    st.error(f"Configuration file {config_path} not found.")
    st.stop()

with open(config_path) as file:
    try:
        config = yaml.load(file, Loader=SafeLoader)
    except yaml.YAMLError as e:
        st.error(f"Error reading configuration file: {e}")
        st.stop()

# Load hashed passwords
file_path = 'hashed_pw.pkl'
if not os.path.exists(file_path):
    st.error(f"Hashed passwords file {file_path} not found.")
    st.stop()

with open(file_path, 'rb') as f:
    try:
        hashed_passwords = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading hashed passwords: {e}")
        st.stop()

# User authentication setup
names = ['Gaurav Kumar']
usernames = ['gauravkumar']

# Create credentials dictionary
credentials = {
    'usernames': {
        'gauravkumar': {
            'email': 'gauravkumar@gmail.com',
            'name': 'Gaurav Kumar',
            'password': hashed_passwords[0]  # Use hashed password
        }
    },
    'cookie': config['cookie'],
    'preauthorized': config.get('preauthorized', {})
}

# Initialize the authenticator with the credentials dictionary and additional arguments
authenticator = stauth.Authenticate(
    credentials,
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login and check authentication status
name, authentication_status, username = authenticator.login(location='main')

# Check user authentication status
if authentication_status == False:
    st.error('Username/Password is incorrect')
    st.stop()

if authentication_status == None:
    st.warning('Please enter username and password')
    st.stop()

if authentication_status:
    st.sidebar.title(f"Welcome {name}")

    # Sidebar for NewsAPI input
    authenticator.logout('Logout', location='sidebar')
    st.sidebar.title('NewsAPI Search')
    news_query = st.sidebar.text_input('Enter search query for news articles')
    fetch_news_clicked = st.sidebar.button('Fetch News Articles')

    # Sidebar for processing URLs and OpenAI integration
    st.sidebar.title('Process Article URLs')
    urls = [st.sidebar.text_input(f'URL {i+1}') for i in range(3)]
    process_url_clicked = st.sidebar.button('Process URLs')

    # File path for saving and loading the FAISS vectorstore
    faiss_file_path = 'faiss_store_openai.pkl'
    main_placeholder = st.empty()
    llm = OpenAI(temperature=0.9, max_tokens=500)

    # Fetch news articles using NewsAPI
    if fetch_news_clicked:
        news_api_key = os.getenv('NEWS_API_KEY')
        if news_api_key:
            url = f'https://newsapi.org/v2/everything?q={news_query}&apiKey={news_api_key}'
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                st.header('News Articles')
                for article in articles:
                    st.subheader(article['title'])
                    st.write(article['description'])
                    st.write(f"[Read more]({article['url']})")
                    
                    # Summarize the article using OpenAI
                    article_text = article['content'] or article['description'] or article['title']
                    summary_prompt = f"Summarize the following article:\n\n{article_text}"
                    summary_response = llm(summary_prompt)

                    # Display the summary
                    if summary_response:
                        st.write("Summary:")
                        st.write(summary_response)
                    else:
                        st.write("Failed to generate summary.")
            else:
                st.error('Failed to fetch news articles')
        else:
            st.error('NewsAPI key not found. Please add it to the .env file.')

    # Process URLs to create embeddings and save them to FAISS index
    if process_url_clicked:
        if any(urls):
            # Load data from URLs
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text('Data Loading...Started...✅✅✅')
            data = loader.load()
            
            # Split data into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text('Text Splitter...Started...✅✅✅')
            docs = text_splitter.split_documents(data)
            
            # Create embeddings and save them to FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text('Embedding Vector Started Building...✅✅✅')
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(faiss_file_path, 'wb') as f:
                pickle.dump((vectorstore_openai.docstore, vectorstore_openai.index_to_docstore_id, vectorstore_openai.index), f)
        else:
            st.warning('Please enter at least one URL.')

    # Input for user's question
    query = main_placeholder.text_input('Question: ')
    if query:
        # Check if the FAISS vectorstore file exists
        if os.path.exists(faiss_file_path):
            # Load the FAISS vectorstore from the pickle file
            with open(faiss_file_path, 'rb') as f:
                docstore, index_to_docstore_id, index = pickle.load(f)

                # Create embeddings instance
                embedding_function = OpenAIEmbeddings()

                # Initialize FAISS vectorstore
                vectorstore = FAISS(docstore=docstore, index_to_docstore_id=index_to_docstore_id, index=index, embedding_function=embedding_function)
                
                # Create the retrieval QA chain
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                
                # Generate the result by passing the user's question to the chain
                result = chain({'question': query}, return_only_outputs=True)
                
                # Display the answer to the user's question
                st.header('Answer')
                st.write(result['answer'])

                # Display sources, if available
                sources = result.get('sources', '')
                if sources:
                    st.subheader('Sources:')
                    sources_list = sources.split('\n')  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)

                # Save query for future reference
                def save_query(query, answer, sources):
                    data = {
                        'query': query,
                        'answer': answer,
                        'sources': sources
                    }
                    with open('queries.json', 'a') as f:
                        json.dump(data, f)
                        f.write("\n")

                save_query(query, result['answer'], sources)
        else:
            st.error('FAISS vectorstore file not found. Please process URLs first.')

    # Export summaries
    def export_summaries():
        if os.path.exists('queries.json'):
            with open('queries.json') as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                df = pd.DataFrame(data)
                df.to_csv('summaries.csv', index=False)
            st.success('Summaries exported successfully')
        else:
            st.error('No queries found to export.')

    if st.button('Export Summaries'):
        export_summaries()

    # View historical data
    def load_queries():
        if os.path.exists('queries.json'):
            with open('queries.json') as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                return data
        else:
            st.error('No historical data found.')
            return []

    if st.button('View Historical Data'):
        data = load_queries()
        if data:
            df = pd.DataFrame(data)
            st.write(df)