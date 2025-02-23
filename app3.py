import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Title and description
st.title("Chat with Your PDF")
st.write("Ask questions about the content of your PDF document.")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set up the embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set the Anthropic API key (replace with your own)
os.environ["ANTHROPIC_API_KEY"] = "enter key"

# Initialize the LLM with Anthropic's chat model
llm = ChatAnthropic(model_name="claude-3-haiku-20240307")  # Updated to a current model

# Function to load and process the PDF
@st.cache_resource
def load_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found at: {pdf_path}")
            return None
        index = VectorstoreIndexCreator(embedding=embedding).from_loaders([PyPDFLoader(pdf_path)])
        return index
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

# Let the user specify the PDF path
pdf_path = st.text_input("Enter PDF file path:", r"C:\Users\MR QUANTUM\Desktop\bg.pdf")
with st.spinner("Loading PDF and creating index..."):
    index = load_pdf(pdf_path)

# Stop if the PDF fails to load
if index is None:
    st.error("Failed to load the PDF. Please check the file path and try again.")
    st.stop()

# Set up the QA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever()
    )
except Exception as e:
    st.error(f"Error creating QA chain: {str(e)}")
    st.stop()

# Chat input for user questions
if prompt := st.chat_input("What would you like to know about the document?"):
    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Generating response..."):
        try:
            response = qa_chain.run(prompt)
            # Add the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])