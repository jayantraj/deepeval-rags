import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import streamlit as st

# loads the environment variables
# load_dotenv()

# Retrieve API keys from environment variables
pinecone_api_key = st.secrets.PINECONE_API_KEY
openai_api_key = st.secrets.OPENAI_API_KEY

# setup openai
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone

pc = Pinecone(
    api_key=pinecone_api_key
)

# Create a Pinecone index if it doesn't exist
index_name = "rags-pdf-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, # should match the output dimensions of the embeddings. we are using: text-embedding-3-small
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to split text into chunks
def split_text(text, max_tokens=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to create openai embeddings
def create_embeddings(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    embeddings = response.data[0].embedding
    return embeddings

# Function to query Pinecone
def query_pinecone(query, top_k=4):
    query_embedding = create_embeddings(query)
    # query_response = index.query(vector=query_embedding, top_k=top_k)
    query_response = index.query(vector=query_embedding, top_k=top_k, include_values=False, include_metadata=True)

    return query_response['matches']

# Function to get response from OpenAI
def get_response_from_openai(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=150)
    return response.choices[0].text.strip()


# Streamlit app
st.title("RAG: Multi-PDF ChatBot")

# Upload multiple PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    st.write("Embedding and Indexing PDFs...")

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary path
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Split text into manageable chunks
        chunks = split_text(text, max_tokens=2000)

        # Step 3: Create embeddings for each chunk and upsert to Pinecone
        for i, chunk in enumerate(chunks):
            embeddings = create_embeddings(chunk)
            metadata = {"text": chunk}
            index.upsert(vectors=[(f"{pdf_path}_chunk_{i}", embeddings, metadata)])
    
    st.write("PDFs processed successfully!")



# Enter query
query = st.text_input("Enter your question:")

if query:
    results = query_pinecone(query)

    if results and results[0]['score'] >= 0.4:  # Adjust threshold as necessary
        context = " ".join([match['metadata']['text'] for match in results if 'metadata' in match and 'text' in match['metadata']])
        response = get_response_from_openai(query, context)
    else:
        response = "I can't answer from the given PDFs."
    
    st.write("Answer:")
    st.write(response)

        # Display the similarity scores
    for match in results:
        score = match['score']
        text = match['metadata']['text']
        st.write(f"Similarity Score: {score}")
        st.write(f"Text: {text}")
        st.write("----")