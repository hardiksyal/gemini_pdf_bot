# gemini_pdf_bot

PDF Chatbot is a Streamlit web application that allows users to interactively ask questions based on the content of uploaded PDF files. The chatbot utilizes language models and question-answering techniques to provide responses to user queries.

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  
## Dependencies
The project relies on the following dependencies:
- [Streamlit](https://streamlit.io/): A Python library for building interactive web applications.
- [PyPDF2](https://pythonhosted.org/PyPDF2/): A library for reading and manipulating PDF files.
- [LangChain](https://github.com/langchain/langchain): A library for natural language processing tasks, including text splitting, document loading, and question answering.
- [Google Generative AI](https://github.com/google-research/google-generativeai): A library for accessing Google's Generative AI models.
- [dotenv](https://pypi.org/project/python-dotenv/): A Python library for reading environment variables from `.env` files.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hardiksyal/gemini_pdf_bot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd gemini_pdf_bot
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project directory:
   ```plaintext
   GOOGLE_API_KEY="your-google-api-key"
   ```
Make sure to replace `your-google-api-key` with your actual Google API key.

## Usage
To run the PDF Chatbot web application, execute the following command:
```bash
streamlit run app.py
```
This will start the Streamlit server, and you can access the application in your web browser.
