from fastapi import FastAPI
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

# 1. Setup Models
embeddings = OllamaEmbeddings(model="qwen2.5-coder:0.5b")
llm = ChatOllama(model="mistral")

# 2. Load and Index HTML Template
loader = BSHTMLLoader("templates/reference.html")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# 3. Create the Chain
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())

@app.get("/generate")
async def generate(prompt: str):
    response = qa_chain.invoke(prompt)
    return {"code": response["result"]}