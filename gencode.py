import chromadb
import requests

def generate_code(user_prompt):
    # 1. Query ChromaDB for relevant context
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection(name="python_templates")
    
    results = collection.query(query_texts=[user_prompt], n_results=1)
    context_code = results['documents'][0][0] if results['documents'] else ""

    # 2. Construct the RAG Prompt
    full_prompt = f"""
    Use the following Python template as a reference:
    ---
    {context_code}
    ---
    Based on the reference above, fulfill this request: {user_prompt}
    Return ONLY valid Python code.
    """

    # 3. Call Ollama (fastapi-gen model created in your docker-compose)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": full_prompt,
            "stream": False
        }
    )
    
    return response.json().get("response")

# Example Usage
prompt = "Create a FastAPI endpoint that returns a user list"
print(generate_code(prompt))