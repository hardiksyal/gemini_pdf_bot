import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda pdf: extract_text_from_pdf(pdf), pdf_docs))
    return "".join(results)

def extract_text_from_pdf(pdf):
    """Helper function to extract text from a single PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate embeddings and create vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(prompt_template, temperature=0.7):
    """Initialize conversational chain with a prompt template."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    """Handle user input and generate response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Retrieve more documents to ensure better context
    
    # Format history into a readable string
    history_str = "\n".join([f"Q: {entry['user_question']}\nA: {entry['answer']}" for entry in chat_history])
    
    # Generalized prompt template for conversational chain
    prompt_template = """
    You are a knowledgeable and insightful AI assistant. Your purpose is to provide comprehensive, detailed, and accurate answers to user questions based on the content of uploaded PDF documents. You are adept at understanding complex topics and can synthesize information from various sections of the document to provide thorough and well-organized responses. Ensure your tone is professional, informative, and clear.
    When responding, use the context provided by the document to form your answers. If the context does not explicitly contain the answer, make a reasonable inference based on related content, but clearly state any assumptions made. If the answer is not found in the context, honestly communicate this to the user. Additionally, provide related information or insights that might be helpful.

    Context:\n{context}\n
    History:\n{history}\n
    Question:\n{question}\n

    Answer:
    """
    
    chain = get_conversational_chain(prompt_template, temperature=0.7)

    context = "\n".join([doc.page_content for doc in docs])
    response = chain({
        "input_documents": docs,
        "context": context,
        "history": history_str,
        "question": user_question
    }, return_only_outputs=True)

    return response["output_text"]

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        chat_history = st.session_state["chat_history"]
        with st.spinner("Generating response..."):
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
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed!")

if __name__ == "__main__":
    main()
