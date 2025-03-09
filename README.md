#  LLM-Powered PDF ChatBot
This project is a Large Language Model (LLM)-Powered PDF ChatBot built using Streamlit, FAISS, LangChain, and OpenAI API. 
The ChatBot allows users to upload PDF documents and ask questions about the content of the document. 
It uses FAISS (Facebook AI Similarity Search) for quick document embeddings search and OpenAI API for generating accurate and contextual responses.

# Features
PDF Upload: Allows users to upload PDF documents.
Ask Questions: Users can ask questions based on the document's content.
FAISS Indexing: Uses FAISS for efficient semantic search on document embeddings.
OpenAI API Integration: Uses OpenAI's LLM (like GPT models) to generate accurate answers.
Local Storage of Embeddings: Caches the embeddings to avoid redundant processing.
Streamlit UI: Provides a simple and interactive web interface for seamless user experience.

# File Structure
LLM-Powered-PDF-ChatBot
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Required packages and dependencies
├── .env                   # API keys and environment variables
├── saved_embeddings/      # Directory to store FAISS index and embeddings (created dynamically)

# Technologies Used
Python 3.10+
Streamlit - For building a web-based user interface.
FAISS - For fast document similarity search.
LangChain - For document chunking and processing.
OpenAI API - For generating contextual answers.
PyPDF2 - For reading PDF documents.
Sentence Transformers - For embedding document content.

# How It Works
Upload a PDF File: The user uploads a PDF file.
Extract Text: The text from the PDF file is extracted and split into smaller chunks.
Generate Embeddings: Each chunk is converted into embeddings using Sentence Transformers.
FAISS Indexing: The embeddings are indexed using FAISS for quick semantic search.
User Query: The user types a question, and the embeddings are searched for contextually relevant chunks.
OpenAI API: The content and query are sent to OpenAI's API to generate an accurate answer.
Cache Embeddings: The embeddings are cached locally to avoid repetitive processing.

#  Requirements File
openai
streamlit
python-dotenv
faiss-cpu
PyPDF2
langchain
sentence-transformers
numpy

# Expected Output
Allows users to upload a PDF file.
Provides answers based on the PDF content.
Caches embeddings to minimize processing time.
Handles API rate limits and errors gracefull

# Future Improvements
Implement multi-file PDF upload support.
Add chatbot history functionality.
Deploy on cloud platforms like Heroku or Vercel.
Integrate with other LLM models like Claude or Gemini.

