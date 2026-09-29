from openai import OpenAI
import chromadb
import os


client = OpenAI()


# ============================================================
# 1. LOAD DOCUMENT
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Document loaded successfully")


# ============================================================
# 2. CHUNKING
# ============================================================

chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):

    chunks.append(
        text[i:i + chunk_size]
    )

print("Number of chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = []

for chunk in chunks:

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embeddings.append(
        response.data[0].embedding
    )

print("Embeddings created")


# ============================================================
# 4. STORE IN CHROMADB
# ============================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="retrieval_demo"
)

collection.add(

    ids=[
        str(i)
        for i in range(len(chunks))
    ],

    documents=chunks,

    embeddings=embeddings
)

print("Documents stored in ChromaDB")


# ============================================================
# 5. USER QUESTION
# ============================================================

question = "What is RAG and how does it work?"

print("\nQUESTION:")
print(question)


# ============================================================
# 6. CREATE QUESTION EMBEDDING
# ============================================================

response = client.embeddings.create(

    model="text-embedding-3-small",

    input=question
)

question_vector = response.data[0].embedding

print("\nQuestion embedding created")


# ============================================================
# 7. RETRIEVAL + TOP-K
# ============================================================

TOP_K = 3

results = collection.query(

    query_embeddings=[
        question_vector
    ],

    n_results=TOP_K
)

retrieved_chunks = results["documents"][0]


print("\n================ RETRIEVED CHUNKS ================")

print("TOP-K:", TOP_K)


for i, chunk in enumerate(retrieved_chunks):

    print("\n----------------")
    print("RESULT:", i + 1)
    print(chunk)


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

context = "\n\n".join(
    retrieved_chunks
)


print("\n================ CONTEXT ================")

print(context)


# ============================================================
# 9. CREATE PROMPT
# ============================================================

prompt = f"""
You are an enterprise AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context,
say:

"I don't know based on the provided knowledge base."

Knowledge Base Context:
-----------------------
{context}
-----------------------

User Question:
{question}
"""


print("\n================ PROMPT ================")

print(prompt)


# ============================================================
# 10. SEND PROMPT TO LLM
# ============================================================

response = client.chat.completions.create(

    model="gpt-4.1-mini",

    messages=[

        {
            "role": "system",
            "content": "You are a helpful enterprise AI assistant."
        },

        {
            "role": "user",
            "content": prompt
        }

    ],

    temperature=0.2,

    top_p=0.9
)


# ============================================================
# 11. FINAL ANSWER
# ============================================================

answer = response.choices[0].message.content


print("\n================ FINAL ANSWER ================")

print(answer)
