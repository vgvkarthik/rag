import os
import time
from langchain_community.document_loaders import csv_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core import exceptions
from diskcache import Cache
import hashlib
from dotenv import load_dotenv


#1. Load document
loader = csv_loader.CSVLoader(file_path="C:\\src\\neuneworks\\rag\\Viewsdata.csv")
documents = loader.load()

#2. Create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#3. Create Embeddings
cache = Cache("./embeddings_cache")
cache.clear()
def get_cache_key(text):
    """Generate a cache key for the text."""
    return hashlib.md5(text.encode()).hexdigest()

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(exceptions.ResourceExhausted)
)
def generate_embeddings_with_retry(text, embeddings_model):
    """Generate embeddings with caching and retry logic."""
    cache_key = get_cache_key(text)
    
    # Try to get from cache first
    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        print("Using cached embedding")
        return cached_embedding
    
    try:
        embedding = embeddings_model.embed_query(text)
        # Store in cache
        cache.set(cache_key, embedding)
        return embedding
    except exceptions.ResourceExhausted as e:
        print(f"Rate limit reached, retrying after backoff...")
        raise e

try:
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_APIKEY")
    )
    
    # Add delay between requests for non-cached embeddings
    for chunk in chunks:
        time.sleep(1)  # 1 second delay between requests
        embedding = generate_embeddings_with_retry(chunk.page_content, embeddings)

except Exception as e:
    print(f"Error: {str(e)}")
    print("Suggestions:")
    print("1. Check your API quota at https://makersuite.google.com/")
    print("2. Clear cache if needed: cache.clear()")
    print("3. Switch to local embeddings if needed")

#4. Store Embeddings in Vector Store
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

retriever = db.as_retriever()

#5. Instruct LLM to reference vector store
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=os.getenv("GEMINI_APIKEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question based on the provided context:\n\n{context}"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What is the main topic of the document?"})
print(response["answer"])
response = rag_chain.invoke({"input": "How many views for Nvidia video?"})
print(response["answer"])

