from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load your documents (adjust path to where your documents are)
# For example, if you have .txt files in a 'documents' folder:
loader = DirectoryLoader('./documents/', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

print(f"Loaded {len(documents)} documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

print(f"Split into {len(texts)} chunks")

# Create FAISS database
print("Creating FAISS database...")
vectordb = FAISS.from_documents(texts, embedding)

# Save the database
vectordb.save_local("./faiss_db")

print("✅ FAISS database created and saved to ./faiss_db")