import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Document loaded")
print()
print(text)
