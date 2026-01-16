import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Standard imports for LangChain 0.2+
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")

# 1. Initialize LangChain Components
embeddings = OllamaEmbeddings(model="qwen2.5-coder:0.5b", base_url=OLLAMA_URL)
llm = ChatOllama(model="fastapi-gen", base_url=OLLAMA_URL)

# 2. Load and Index the Reference HTML
loader = BSHTMLLoader("templates/reference.html")
docs = loader.load()
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. Setup RAG Chain
prompt = ChatPromptTemplate.from_template("""
Use the following context to generate Python code:
{context}
User Request: {input}
Answer in valid Python code only.
""")
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate")
async def generate(user_input: str):
    response = retrieval_chain.invoke({"input": user_input})
    return {"code": response["answer"]}