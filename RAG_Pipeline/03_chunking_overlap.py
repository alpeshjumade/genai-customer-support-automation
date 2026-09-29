from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")


with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


chunks = splitter.split_text(text)


print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print("\n====================")
    print("CHUNK", i)
    print("====================")

    print(chunk)
