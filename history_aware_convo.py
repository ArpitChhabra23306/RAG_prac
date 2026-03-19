import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"

# 1. Use the SAME embedding model used in ingestion_pipeline.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory, 
    embedding_function=embeddings
)

# 2. Set up the Generative AI model (Groq)
model = ChatGroq(model="llama-3.3-70b-versatile")

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone so it's better for search
        history_prompt = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question text, nothing else."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(history_prompt)
        search_question = result.content.strip()
        print(f"Standalone Search Query: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents using the standalone question
    # We use k=3 to keep the prompt small and fast
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant document chunks.")
    
    # Step 3: Create final context-aware prompt
    context = "\n".join([f"- {doc.page_content}" for doc in docs])
    
    combined_input = f"""Based on the following document context and our chat history, please answer the user's question.

Context:
{context}

User Question: {user_question}

Please provide a clear, helpful answer using ONLY the provided context. If the answer is not in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    # Step 4: Get the answer from Groq
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    print("Thinking... (Consulting Groq Cloud)")
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Update conversation history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nAI Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("\n🚀 RAG Chatbot is Ready! (Groq + Local Search)")
    print("Ask me anything about your documents. Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\nYour question: ")
            
            if not question.strip():
                continue
                
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            ask_question(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    start_chat()
