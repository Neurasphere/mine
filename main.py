import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings  # Import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

# Load the environment variables from the .env file
load_dotenv()

# Get the API key from the environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
print(openai_api_key)

# Use the API key in the OpenAI initialization
openai.api_key = openai_api_key

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Initialize OpenAIEmbeddings

# Load image
img_path = r"C:\Users\HC\PycharmProjects\pythonProject1\neurachat_logo.png"
img = Image.open(img_path)

# Resize the image
base_width = 700
w_percent = base_width / float(img.width)
h_size = int(float(img.height) * float(w_percent))
img = img.resize((base_width, h_size), Image.LANCZOS)

st.image(img, width=700)

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Initialize or fetch the conversation history from the session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Use OpenAI embeddings
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        if 'query_value' not in st.session_state:
            st.session_state.query_value = ""

        query = st.text_input("Ask questions about your PDF file:", value=st.session_state.query_value, key="unique_key_for_clearing_input")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Pass the openai_api_key to the OpenAI instance
            llm = OpenAI(openai_api_key=openai_api_key)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Append to the conversation history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", response))

            # Display conversation history
            for entry in st.session_state.chat_history:
                if entry[0] == "You":
                    st.write(f"You: {entry[1]}")
                else:
                    st.write(f"Bot: {entry[1]}")

            # Reset the input field's value
            st.session_state.query_value = ""
        else:
            st.session_state.query_value = query  # This retains the input value in the session state

if __name__ == '__main__':
    main()
