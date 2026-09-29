from openai import OpenAI
import chromadb
import os


client = OpenAI()


# Load document

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# Chunk

chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):

    chunks.append(
        text[i:i + chunk_size]
    )


# Embeddings

embeddings = []

for chunk in chunks:

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embeddings.append(
        response.data[0].embedding
    )


# ChromaDB

chroma_client = chromadb.Client()


collection = chroma_client.get_or_create_collection(
    name="practice_collection"
)


collection.add(

    ids=[
        str(i)
        for i in range(len(chunks))
    ],

    documents=chunks,

    embeddings=embeddings
)


print("Stored successfully!")

print(
    "Total documents:",
    collection.count()
)
