import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
from PyPDF2 import PdfReader # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain.text_splitter import CharacterTextSplitter # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from langchain.chains import ConversationalRetrievalChain # type: ignore
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.0,  # Adjust as needed
    api_key="",
)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    print("User  question:", user_question)  # Debug
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation chain not initialized.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                st.write("Raw text extracted:", raw_text)  # Debug statement

                if raw_text.strip():  # Check if raw_text is not empty
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write("Text chunks generated:", text_chunks)  # Debug statement

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    print("Vector store created:", vectorstore is not None)  # Debug

                    if vectorstore:  # Check if vectorstore is created
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Conversation chain initialized!")  # Success message
                    else:
                        st.error("Vector store could not be created.")
                else:
                    st.error("No text extracted from PDFs. Please check the files.")

if __name__ == '__main__':
    main()
