import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

def setup_qa_system(uploaded_file):
    # Create the temp directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded file to the temp directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load documents from the file
    loader = TextLoader(temp_file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Initialize the embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save documents to ChromaDB
    db2 = Chroma.from_documents(docs, embedding=embedding, persist_directory="./chroma_db")

    # Load documents from ChromaDB
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    retriever = db3.as_retriever()

    # Initialize the LLM
    llm = Ollama(model="llama3")

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain

def answer_query(qa_chain, query):
    response = qa_chain({"query": query})
    return response["result"]
