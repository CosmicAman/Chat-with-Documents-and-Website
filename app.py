import streamlit as st
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
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS

load_dotenv()

def get_pdf_text(pdf_files): # Function to extract text from PDF files
    text = ""
    try:
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error while extracting text from PDF: {e}")
    finally:
        return text

def get_text_chunks(text):
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
        text_chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
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

def create_conversation_chain(vectorstore): #Function to create conversation chain based on vector store
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

def main(): #Main
    st.set_page_config(page_title="Chat with documents and websites", page_icon="👾")


    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap'); 

.title {
  font-family: 'Bebas Neue', cursive; 
  font-size: 45px;
  color: #958F9A ;
  text-align: center;
  transition: text-shadow 0.3s ease-in-out; 
}

.title:hover {
  text-shadow: 0 0 45px #6C02B9 ;
}
</style>

<h1 class='title'>Chat with documents and websites 🙋‍♂️</h1>
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

    conversation_chain = create_conversation_chain(vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

    user_query = st.chat_input("Type your message here...")
    if user_query:
        try:
            # Check if the query is relevant to the provided content
            if is_query_relevant(user_query, vectorstore):
                response = conversation_chain.invoke({"chat_history": st.session_state.chat_history, "input": user_query})
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response['answer']))
            else:
                st.error("Sorry, I can only respond to questions related to the provided content.")
        except Exception as e:
            st.error(f"Error while processing user query: {e}")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.markdown(
                f"<div style='background-color: #3b3a39; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>Rudra🤖<br /> <br /> {message.content}</div>",
                unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.markdown(
                f"<div style='background-color: #935DC6 ; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>You🙋<br /> <br /> {message.content}</div>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
