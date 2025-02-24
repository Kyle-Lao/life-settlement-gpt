import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import sys

# Print a confirmation that the script is running
print("✅ store_texts.py script is running...")

# Check if OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    print("❌ ERROR: OPENAI_API_KEY is not set.")
    print("➡️  Set it using: export OPENAI_API_KEY='your_openai_api_key_here'")
    sys.exit(1)

# Initialize ChromaDB
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())

# Folder containing extracted text files
text_folder = "texts"

# Ensure the folder exists
if not os.path.exists(text_folder):
    print(f"❌ ERROR: The folder '{text_folder}' does not exist. Run extract_text.py first.")
    sys.exit(1)

# Split text for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# Load and store text files in ChromaDB
for txt_file in os.listdir(text_folder):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(text_folder, txt_file)

        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            db.add_documents(chunks)
            print(f"✅ Stored: {txt_file}")
        except Exception as e:
            print(f"❌ ERROR: Failed to process {txt_file}: {e}")

print("🎉 ✅ All texts stored in ChromaDB successfully!")

