"""
Text file summarization using HuggingFace's FREE Inference API
Updated with new Inference Providers API endpoint
No local models, no PyTorch issues, no payment required!
"""

import requests
import time
import os

# Get your API token from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

def summarize_with_api(text, max_length=150, min_length=30):
    """
    Summarize text using HuggingFace's NEW Inference Providers API
    """
    # Updated API endpoint
    API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 503:
        # Model is loading, wait and retry
        print("Model is loading... waiting 20 seconds...")
        time.sleep(20)
        response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def summarize_text_file(file_path, max_chunk_size=1000):
    """
    Summarize a text file, handling long documents by chunking
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text: {len(text)} characters")
    
    # Split into chunks if needed (API has token limits)
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    print(f"Processing {len(chunks)} chunk(s)...\n")
    
    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        try:
            summary = summarize_with_api(chunk)
            summaries.append(summary)
            print(f"✓ Chunk {i+1} done")
            time.sleep(1)  # Be nice to the free API
        except Exception as e:
            print(f"✗ Error on chunk {i+1}: {e}")
            summaries.append(f"[Error summarizing chunk {i+1}]")
    
    # Combine summaries
    final_summary = ' '.join(summaries)
    
    # If multiple chunks, do final pass
    if len(chunks) > 1 and len(final_summary) > 1000:
        print("\nCreating final summary...")
        try:
            final_summary = summarize_with_api(final_summary)
        except Exception as e:
            print(f"Final summary failed: {e}")
    
    return final_summary

def summarize_text_string(text):
    """
    Summarize a text string directly
    """
    return summarize_with_api(text)

# Example usage
if __name__ == "__main__":
    # Check if API token is set
    if not HUGGINGFACE_API_TOKEN:
        print("⚠️  Please set your HuggingFace API token!")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a free account (if needed)")
        print("3. Create a new token (read access is enough)")
        print("4. Set environment variable: HUGGINGFACEHUB_API_TOKEN=your_token_here")
        print("   Or create a .env file with: HUGGINGFACEHUB_API_TOKEN=your_token_here")
        exit(1)
    
    # Test with example text first
    example_text = """
    Artificial intelligence has made remarkable progress in recent years, 
    transforming various industries and aspects of daily life. Machine learning 
    models, particularly deep neural networks, have achieved unprecedented 
    performance in tasks such as image recognition, natural language processing, 
    and game playing. The advent of large language models has revolutionized 
    how we interact with computers, enabling more natural and intuitive 
    communication. However, these advances also raise important questions about 
    ethics, bias, and the future of work. As AI systems become more capable, 
    society must grapple with ensuring they are developed and deployed responsibly, 
    with appropriate safeguards and oversight.
    """
    
    print("🧪 Testing summarization with example text...\n")
    try:
        summary = summarize_text_string(example_text)
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(summary)
        print("\n" + "="*60)
        
        # Now try summarizing a file if it exists
        file_path = "fashion.txt"
        if os.path.exists(file_path):
            print(f"\n📄 Summarizing file: {file_path}")
            file_summary = summarize_text_file(file_path)
            print("\n" + "="*60)
            print("FILE SUMMARY:")
            print("="*60)
            print(file_summary)
            
            # Save the summary
            with open("summary_output.txt", "w") as f:
                f.write(file_summary)
            print(f"\n💾 Summary saved to: summary_output.txt")
        else:
            print(f"\n⚠️  File '{file_path}' not found. Skipping file summarization.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your API token is valid")
        print("2. Make sure you have internet connection")
        print("3. The model might be temporarily unavailable")