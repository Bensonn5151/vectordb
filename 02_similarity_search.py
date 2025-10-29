from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# This will use your local CPU - slower but no API issues
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = FAISS.load_local("./faiss_db", embedding, allow_dangerous_deserialization=True)

# Simple retrieval without LLM (just semantic search)
print("🤖 PIPEDA Knowledge Base - Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    
    if question.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye! 👋")
        break
    
    # Just do semantic search
    docs = vectordb.similarity_search(question, k=3)
    
    print("\n📚 Most relevant information:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Result {i} ---")
        content = doc.page_content.strip()[:400]
        print(content)
        if len(doc.page_content) > 400:
            print("...")
        print()
    print("-" * 60 + "\n")