from openai import OpenAI
import os

client = OpenAI()


# Load document

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# Chunking

chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):

    chunks.append(
        text[i:i + chunk_size]
    )


# Embedding

embedding_model = "text-embedding-3-small"

embeddings = []


for chunk in chunks:

    response = client.embeddings.create(
        model=embedding_model,
        input=chunk
    )

    vector = response.data[0].embedding

    embeddings.append(vector)


# Display

for i in range(len(chunks)):

    print("\n===================")

    print("CHUNK:", i)

    print(chunks[i])

    print("\nVECTOR SIZE:")

    print(len(embeddings[i]))

    print("\nFIRST 5 VALUES:")

    print(embeddings[i][:5])
