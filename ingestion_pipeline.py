import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
print("Imports were successful! Pipeline is ready (using Cohere).")


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter for better uniformity"""
    print("Splitting documents into chunks...")
    
    # We use RecursiveCharacterTextSplitter for better semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        print(f"Split into {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Length: {len(chunk.page_content)} characters")
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using Cohere Cloud Embeddings"""
    print("Creating embeddings and storing in ChromaDB via Cohere Cloud...")

    # Delete old DB to prevent dimension mismatch and duplicates
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}...")
        shutil.rmtree(persist_directory)

    # Use Cohere Cloud Embeddings (1024 dimensions)
    embedding_model = CohereEmbeddings(model="embed-english-v3.0")
    
    # Create ChromaDB vector store
    print("--- Creating vector store (Cloud Math) ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    # 1. Load documents
    documents = load_documents(docs_path="docs")
    # 2. Split documents
    chunks = split_documents(documents)
    # 3. Create vector store
    create_vector_store(chunks)

if __name__ == "__main__":
    main()

