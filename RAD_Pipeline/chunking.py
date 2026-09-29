import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# CHUNKING

chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):

    chunk = text[i:i + chunk_size]

    chunks.append(chunk)


print("Total chunks:", len(chunks))


# Print every chunk

for i, chunk in enumerate(chunks):

    print("\n====================")
    print("CHUNK", i)
    print("====================")

    print(chunk)
