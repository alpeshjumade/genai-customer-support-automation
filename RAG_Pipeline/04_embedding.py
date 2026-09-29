from openai import OpenAI

client = OpenAI()


text = "RAG combines retrieval with a large language model."


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)


embedding = response.data[0].embedding


print("Original text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nVector dimensions:")
print(len(embedding))
