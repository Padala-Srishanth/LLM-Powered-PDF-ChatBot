import openai  # Import the official OpenAI library for error handling
import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.schema import Document
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Streamlit sidebar
with st.sidebar:
    st.title('LLM ChatBot')
    st.markdown('''
    ### About 
    This app is an LLM-Powered ChatBot
    ''')


def create_faiss_index(embeddings, texts):
    """Creates a FAISS index from embeddings and saves the text data."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embeddings)  # Add embeddings to the index
    return index, texts


def get_local_embeddings(chunks): 
    """Function to get embeddings using Sentence Transformers."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        embeddings = model.encode(chunks)
        return np.array(embeddings), chunks
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None, None


def search_faiss_index(query, index, texts, model, k=3):
    """Searches the FAISS index for the closest matches to the query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [Document(page_content=texts[i]) for i in indices[0] if i != -1]


def make_api_request(docs, query, retries=3):
    for i in range(retries):
        try:
            # Use the OpenAI client for Grok API
            messages = [{"role": "system", "content": "You are Grok, an intelligent assistant."},
                        {"role": "user", "content": f"Answer the following based on these documents: {query}"}]
            response = client.chat.completions.create(
                model="grok-beta",
                messages=messages
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            wait_time = 2 ** i  # Exponential backoff
            st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.OpenAIError as e:
            st.error(f"API Error: {e}")
            break
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
            break

    st.error("Max retries reached. Please check your API quota or the query.")
    return None


# Cache API responses to avoid duplicate requests
@lru_cache(maxsize=100)
def cached_request(docs_tuple, query):
    docs = [Document(page_content=doc)
            for doc in docs_tuple]  # Convert to Document objects again
    return make_api_request(docs, query)


def main():
    st.header('Chat with your PDF.........')

    pdf = st.file_uploader("Upload your PDF Here: ", type='pdf')

    if pdf is not None:
        st.write(pdf.name)

        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Check if stored embeddings exist
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}_index.pkl") and os.path.exists(f"{store_name}_texts.pkl"):
            with open(f"{store_name}_index.pkl", "rb") as f:
                index = pickle.load(f)
            with open(f"{store_name}_texts.pkl", "rb") as f:
                texts = pickle.load(f)
            st.write('Embeddings loaded from the Disk')
        else:
            # Try to create embeddings using Sentence Transformers
            embeddings, texts = get_local_embeddings(chunks)
            if embeddings is not None:
                index, texts = create_faiss_index(embeddings, texts)

                # Save index and texts to disk
                with open(f"{store_name}_index.pkl", "wb") as f:
                    pickle.dump(index, f)
                with open(f"{store_name}_texts.pkl", "wb") as f:
                    pickle.dump(texts, f)
                st.write("Embeddings stored to disk.")
            else:
                st.error("Unable to create embeddings.")

        # Accept user questions / Query
        query = st.text_input("Ask questions about your PDF")

        if query:
            # Use FAISS to find relevant chunks
            # Model used to encode query
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                docs = search_faiss_index(query, index, texts, model, k=3)

            # Convert docs to tuple of strings (hashable)
                docs_tuple = tuple([doc.page_content for doc in docs])

            # Use the cached request function to avoid duplicate API calls
                response = cached_request(docs_tuple, query)

            except Exception as e:
                st.error(f"Error 3: {e}")
                response = None

            if response:
                st.write("Response:", response)
            else:
                st.write("Failed to retrieve response after retries.")

    else:
        st.write("Please upload a PDF file to proceed.")


if __name__ == '__main__':
    main()
