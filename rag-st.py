# Install necessary libraries
# pip install langchain langchain-community sentence-transformers chromadb ollama

import os
import shutil
from langchain.document_loaders import csv_loader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
#from langchain.llms import Ollama # Example for a local LLM
from langchain_community.llms import CTransformers # Example for a local LLM

chroma_dir = "./chroma_db"
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)

# 1. Load and Chunk Data
loader = csv_loader.CSVLoader(file_path="C:\\src\\neuneworks\\rag\\Viewsdata.csv")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2. Generate Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Store in ChromaDB
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
vectordb.persist()
retriever = vectordb.as_retriever()
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question based on the provided context:\n\n{context}"),
    ("human", "{input}")
])

# 4. Retrieval and 5. LLM Integration
llm = CTransformers(model="C:\\src\\neuneworks\\rag\\mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral")

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "How many views for Nvidia video in the viewdata csv?"
response = qa_chain.invoke({"query": query})
print(response["result"])