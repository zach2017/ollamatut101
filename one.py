import os
import chromadb
import ollama

# --- Configuration ---
# Use HttpClient to connect to your ChromaDB server on port 8000
CHROMA_HOST = "localhost" 
CHROMA_PORT = 8000
COLLECTION_NAME = "mistral_codebase"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
CHUNK_SIZE = 1500  

# Initialize the Remote Client
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_chunks(text, size):
    """Splits a string into smaller chunks."""
    return [text[i:i + size] for i in range(0, len(text), size)]

def index_code(directory):
    # Added 'code_db_mistral' to ignored_dirs just in case local files exist
    ignored_dirs = {'.git', 'venv', '__pycache__', 'code_db_mistral'}
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                print(f"Processing: {path}")
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    chunks = get_chunks(content, CHUNK_SIZE)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{path}_chunk_{i}"
                        
                        # Get embedding from Ollama
                        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                        
                        # Add to the remote collection
                        collection.add(
                            ids=[chunk_id],
                            embeddings=[response["embedding"]],
                            documents=[chunk],
                            metadatas=[{"source": path, "chunk_index": i}]
                        )
                except Exception as e:
                    print(f"Failed to index {path}: {e}")

def query_code(user_query):
    query_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_query)["embedding"]
    
    # Query the remote server for the top 3 most relevant chunks
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    
    context = "\n---\n".join(results["documents"][0])
    
    prompt = f"[INST] Use the code snippets below to answer: {user_query}\n\nContext:\n{context} [/INST]"
    
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]

if __name__ == "__main__":
    # Index current directory to the remote DB
    index_code("./")
    
    # Test a query
    print("\nResult:", query_code("How is the main entry point structured?"))