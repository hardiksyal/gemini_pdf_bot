# Chatbot for PDF Analysis Documentation

## Overview

This documentation provides an overview of the Chatbot for PDF Analysis application, which utilizes natural language processing (NLP) techniques to extract information from PDF documents and provide answers to user queries. The application leverages the Gemini API for NLP tasks and Streamlit for the user interface.

## Dependencies

The application relies on several libraries and dependencies for its functionality:

1. **Streamlit**: Streamlit is a Python library used for building interactive web applications. It simplifies the process of creating user interfaces for data analysis and machine learning models.

```python
import streamlit as st
```

2. **PyPDF2**: PyPDF2 is a Python library for working with PDF files. It allows the application to extract text content from PDF documents, which is essential for processing and analyzing the documents.

```python
from PyPDF2 import PdfReader
```

3. **langchain**: langchain is a library that provides various NLP functionalities, including text splitting, embeddings, vector stores, and conversational chains. It serves as the backbone for the application's NLP capabilities.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
```

4. **GoogleGenerativeAI**: GoogleGenerativeAI is a Python package that interfaces with Google's Generative AI models. It enables the application to generate responses to user queries using advanced language models.

```python
import google.generativeai as genai
```

5. **dotenv**: dotenv is a Python library for parsing `.env` files and loading environment variables. It facilitates the management of sensitive information, such as API keys, by keeping them separate from the codebase.

```python
from dotenv import load_dotenv
```

## Code Explanation

The application consists of several components:

1. **User Interface**: The user interface is built using Streamlit. It provides a simple and intuitive interface for users to upload PDF files, ask questions, and view responses from the chatbot.

```python
# User Interface
st.set_page_config("Chat PDF")
st.header("Chat with PDF")
```

2. **Text Processing**: The `get_pdf_text` function extracts text content from uploaded PDF files using PyPDF2. The `get_text_chunks` function splits the extracted text into manageable chunks to facilitate processing.

```python
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```

3. **Vectorization**: The `get_vector_store` function generates embeddings for the text chunks using GoogleGenerativeAI and creates a vector store using FAISS. This vector store allows for efficient similarity search and retrieval of relevant documents.

```python
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
```

4. **Conversational Chain**: The `get_conversational_chain` function initializes a conversational chain using langchain. It defines a prompt template that provides context, history, and user questions to the chatbot. This chain processes user queries and generates responses based on the provided context.

```python
def get_conversational_chain(prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
```

5. **User Input Handling**: The `user_input` function handles user input by retrieving relevant documents from the vector store and passing them to the conversational chain. It returns the generated response to be displayed in the user interface.

```python
def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    prompt_template = """
    You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional, informative and detailed.Make sure to carefully format your answers in readble and presentable format. If you don't know the answer just say you cannot answer as it's not in the context, don't try to makeup an answer but try your best to look for teh answer in the context again.\n\n
    Context:\n {context}?\n
    History:\n {history}\n
    Question:\n {question}\n

    Answer:
    """
    chain = get_conversational_chain(prompt_template)

    response = chain({"input_documents": docs, "context": "", "history": chat_history, "question": user_question},
                     return_only_outputs=True)

    return response["output_text"]
```

6. **Main Functionality**: The `main` function serves as the entry point for the application. It defines the Streamlit user interface, including text inputs, file uploaders, and buttons. It also manages session state to maintain conversation history across interactions.

```python
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        chat_history = st.session_state.get("chat_history", [])
        answer = user_input(user_question, chat_history)
        st.write("Reply: ", answer)

        # Update chat history
        chat_history.append({"user_question": user_question, "answer": answer})
        st.session_state["chat_history"] = chat_history

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
```

## Conclusion

The Chatbot for PDF Analysis application provides a convenient solution for extracting information from PDF documents and answering user queries. By leveraging NLP techniques and advanced language models, the application offers a user-friendly interface and delivers accurate and informative responses. Its modular design allows for easy extension and customization to meet specific use cases and requirements.

---

## Demonstration

### Uploading PDF Files
![image](https://github.com/hardiksyal/gemini_pdf_bot/assets/63895326/00180c3f-8613-4c56-9d35-7b5d758bf2b3)

*Figure 1: Uploading PDF files to the PDF Chatbot web application.*

### Asking Questions
![image](https://github.com/hardiksyal/gemini_pdf_bot/assets/63895326/c0e79a39-cc1b-429c-9b5e-1de96f6df534)

*Figure 2: Asking questions based on the content of uploaded PDF files.*

### Answering out-of-context questions
![image](https://github.com/hardiksyal/gemini_pdf_bot/assets/63895326/7ff8a9ea-2be8-4474-95d6-ac60e5122d1d)

*Figure 3: Receiving responses from the PDF Chatbot based on the user's questions.*
