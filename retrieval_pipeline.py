from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# 1. Use Cohere Cloud Embeddings (Must match ingestion_pipeline.py)
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "what is meta"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

# Deduplicate results by content (safety net)
seen = set()
unique_docs = []
for doc in relevant_docs:
    if doc.page_content not in seen:
        seen.add(doc.page_content)
        unique_docs.append(doc)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(unique_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"




# 1. Prepare the context from unique documents
context = "\n\n".join([doc.page_content for doc in unique_docs])

# 2. Construct the prompt
combined_input = f"""Based on the following document context, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a clear, helpful answer based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information to answer that question based on the provided documents."
"""

# 3. Create a ChatGroq model
# We'll use llama-3.3-70b-versatile as it's very capable and fast on Groq
model = ChatGroq(model="llama-3.3-70b-versatile")

# 4. Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant specialized in company research."),
    HumanMessage(content=combined_input),
]

# 5. Invoke the model with the combined input
print("\nThinking... (Both Search and AI are in the Cloud now!)")
result = model.invoke(messages)

# 6. Display the generated response
print("\n--- Generated AI Response ---")
print(result.content)