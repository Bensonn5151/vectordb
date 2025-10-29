from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
import dotenv
dotenv.load_dotenv()

import traceback

# Set your Groq API key
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
os.environ["GROQ_API_KEY"] =  os.getenv("GROQ_API_KEY")

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS database
print("Loading FAISS database...")
vectordb = FAISS.load_local(
    "./faiss_db", 
    embedding,
    allow_dangerous_deserialization=True
)
print(f"✅ Loaded FAISS database with {vectordb.index.ntotal} vectors")

# Initialize Groq client
try:
    client = Groq()
    print("✅ Groq client initialized successfully with llama-3.3-70b-versatile")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    exit(1)

def get_llm_response(messages, temperature=0.5, max_tokens=512, stream=False):
    """Get response from Groq LLM using direct API call"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=stream,
            stop=None
        )
        
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            return response
        else:
            return completion.choices[0].message.content
            
    except Exception as e:
        print(f"❌ Error calling Groq API: {e}")
        return None

def get_retrieved_context(question, k=3):
    """Retrieve relevant documents from FAISS"""
    try:
        docs = vectordb.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        source_docs = docs
        return context, source_docs
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return "", []

print("🤖 Chatbot is ready! Type 'quit' to exit.\n")

# Simple chat interface
while True:
    question = input("You: ")
    
    if question.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye! 👋")
        break
    
    if not question.strip():
        continue
    
    try:
        # Step 1: Retrieve relevant documents
        print("🔍 Searching documents...")
        context, source_docs = get_retrieved_context(question, k=3)
        
        # Step 2: Create prompt with context
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, please say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 3: Get LLM response
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("🤖 Generating response...")
        answer = get_llm_response(messages, temperature=0.5, max_tokens=512, stream=False)
        
        if answer:
            print(f"\n🤖 Bot: {answer}\n")
            
            # Show which chunks were used
            if source_docs:
                print("📚 Sources used:")
                for i, doc in enumerate(source_docs, 1):
                    preview = doc.page_content[:150].replace('\n', ' ')
                    print(f"  {i}. {preview}...")
                    if doc.metadata:
                        source_file = doc.metadata.get('source', 'Unknown')
                        print(f"     📄 From: {source_file}")
                print()
        else:
            print("❌ Failed to get response from LLM\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        print("Full traceback:")
        traceback.print_exc()
        print("\nPlease try rephrasing your question.\n")