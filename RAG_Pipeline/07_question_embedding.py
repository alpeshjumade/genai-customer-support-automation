from openai import OpenAI

client = OpenAI()


question = "What is RAG?"


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)


question_vector = response.data[0].embedding


print("Question:")
print(question)

print("\nVector size:")
print(len(question_vector))

print("\nFirst 5 values:")
print(question_vector[:5])
