# Push my vector to Vector database
import os
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
index_name = "chatbot"


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


# Creating Embeddings for Each of The Text Chunks & storing
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# If we already have an index we can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)
