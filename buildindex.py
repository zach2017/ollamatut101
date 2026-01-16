import chromadb

# Initialize Chroma client
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_or_create_collection(name="python_templates")

# Example Python coding patterns to store
coding_examples = [
    {
        "id": "fastapi_get",
        "code": "@app.get('/')\nasync def read_root():\n    return {'Hello': 'World'}",
        "description": "Standard FastAPI GET route definition"
    },
    {
        "id": "pydantic_model",
        "code": "class User(BaseModel):\n    id: int\n    name: str",
        "description": "Pydantic model for data validation"
    }
]

# Add to database
collection.add(
    documents=[ex["code"] for ex in coding_examples],
    metadatas=[{"desc": ex["description"]} for ex in coding_examples],
    ids=[ex["id"] for ex in coding_examples]
)

print("✅ Successfully indexed coding examples in ChromaDB.")