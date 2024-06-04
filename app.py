import streamlit as st
import streamlit.components.v1 as com
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import logging

st.set_page_config(page_title="Chat with Documents and Websites", page_icon="👾")

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# MongoDB connection
uri = "mongodb+srv://aman7480nano:aman@cluster0.xvjyq3x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Sharansh']
chat_history_collection = db['Chat_History']

def save_to_mongodb(user_message, ai_response):
    chat_history_collection.insert_one({"user_message": user_message, "ai_response": ai_response})

def load_from_mongodb(): # Function to load chat history from MongoDB
    return [(doc['user_message'], doc['ai_response']) for doc in chat_history_collection.find()]
try:
    client.admin.command("ping")
    logging.info("Connected to MongoDB successfully!")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")

def get_pdf_text(pdf_files): # Function to extract text from PDF files
    text = ""
    try:
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error while extracting text from PDF: {e}")
    finally:
        return text

def get_text_chunks(text): # Function to split text into chunks
    chunks = []
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error while splitting text into chunks: {e}")
    finally:
        return chunks

def get_vectorstore_from_pdf(pdf_files): # Function to create a vector store from text chunks
    try:
        raw_text = get_pdf_text(pdf_files)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        else:
            st.error("No text found in the provided PDF files.")
    except Exception as e:
        st.error(f"Error while creating vector store from PDF: {e}")

def get_vectorstore_from_url(url): # Function to create a vector store from a website URL
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error while creating vector store from URL: {e}")

def create_conversation_chain(vectorstore): # Function to create conversation chain based on vector store
    try:
        llm = ChatOpenAI()
        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        conversation_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
        return conversation_chain
    except Exception as e:
        st.error(f"Error while creating conversation chain: {e}")

def is_query_relevant(query, vectorstore):
    return True  # Placeholder logic, adjust as needed

def main():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap'); 
            .title {
                font-family: 'Bebas Neue', cursive; 
                font-size: 45px;
                color: #958F9A;
                text-align: center;
                transition: text-shadow 0.3s ease-in-out;   
            }
            .title:hover {
                text-shadow: 0 0 45px #6C02B9;
            }
            .chat-container {
                max-width: 700px;
                margin: 0 auto;
            }
            .chat-bubble {
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-size: 16px;
                line-height: 1.6;
                word-wrap: break-word;
            }
            .human-message {
                background-color: #935DC6;
                color: white;
                text-align: right;
                margin-left: auto;
            }
            .ai-message {
                background-color: #3b3a39;
                color: white;
                text-align: left;
                margin-right: auto;
            }
            .chat-history {
                background-color: #292B2C;
                padding: 10px;
                border-radius: 10px;
                color: white;
                font-size: 14px;
                margin-bottom: 10px;
            }
            .chat-history h4 {
                color: #A0A0A0;
                margin: 5px 0;
                font-size: 16px;
            }
            .chat-history .chat-date {
                margin-bottom: 5px;
                font-weight: bold;
            }
            .chat-history .chat-item {
                margin-left: 10px;
                margin-bottom: 5px;
            }
        </style>
        <h1 class='title'>SAARANSH</h1>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        chat_mode = st.radio("Choose chat mode:", ("Chat with Website", "Chat with PDFs"))
        if chat_mode == "Chat with Website":
            website_url = st.text_input("Website URL")
            pdf_files = None
        else:
            website_url = None
            st.subheader("Upload PDFs")
            pdf_files = st.file_uploader("Upload your PDFs here", type=['pdf'], accept_multiple_files=True)
            if st.button("Submit"):
                if not pdf_files:
                    st.error("Please upload PDF files")
                    return

        # Display chat history
        st.subheader("Chat History")
        chat_history = load_from_mongodb()
        if chat_history:
            today_chats = []
            yesterday_chats = []
            last_7_days_chats = []
            last_30_days_chats = []

            # Organize chat history by date
            from datetime import datetime, timedelta

            now = datetime.now()
            yesterday = now - timedelta(days=1)
            for user_message, ai_response in chat_history:
                chat_time = datetime.now()  # Use the actual timestamp of the message if available
                if chat_time.date() == now.date():
                    today_chats.append((user_message, ai_response))
                elif chat_time.date() == yesterday.date():
                    yesterday_chats.append((user_message, ai_response))
                elif chat_time.date() > now.date() - timedelta(days=7):
                    last_7_days_chats.append((user_message, ai_response))
                elif chat_time.date() > now.date() - timedelta(days=30):
                    last_30_days_chats.append((user_message, ai_response))

            def display_chats(chats, date_label): # Function to display chat history
                if chats:
                    st.write(date_label)
                    for user_message, ai_response in reversed(chats):
                        st.markdown(f"<div class='chat-bubble human-message'><strong>You🙋‍♂️:</strong> {user_message}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='chat-bubble ai-message'><strong>Rudra🤖:</strong> {ai_response}</div>", unsafe_allow_html=True)

            display_chats(today_chats, "Today")
            display_chats(yesterday_chats, "Yesterday")
            display_chats(last_7_days_chats, "Previous 7 Days")
            display_chats(last_30_days_chats, "Previous 30 Days")
        else:
            st.write("No chat history available.")

    if chat_mode == "Chat with Website":
        if not website_url:
            st.error("Please enter a website URL")
            return
        vectorstore = get_vectorstore_from_url(website_url)
    else:
        if not pdf_files:
            st.error("Please upload PDF files")
            return
        vectorstore = get_vectorstore_from_pdf(pdf_files)

    if vectorstore:
        conversation_chain = create_conversation_chain(vectorstore)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

        user_query = st.chat_input("Type your message here...")
        if user_query:
            try:
                if is_query_relevant(user_query, vectorstore):
                    response = conversation_chain.invoke({"chat_history": st.session_state.chat_history, "input": user_query})
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=response['answer']))
                    save_to_mongodb(user_query, response['answer'])
                else:
                    st.error("Sorry, I can only respond to questions related to the provided content.")
            except Exception as e:
                st.error(f"Error while processing user query: {e}")

        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in (st.session_state.chat_history):
            if isinstance(message, AIMessage):
                st.markdown(
                    f"<div class='chat-bubble ai-message'><strong>Rudra🤖:</strong> {message.content}</div>",
                    unsafe_allow_html=True)
            elif isinstance(message, HumanMessage):
                st.markdown(
                    f"<div class='chat-bubble human-message'><strong>You:</strong> {message.content}</div>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
