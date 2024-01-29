from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PINECONE_API_KEY = "1c41f633-451d-4743-bbfa-fc68971f009a"
PINECONE_API_ENV = "gcp-starter"

index_name = "chatbot"


# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls = PyPDFLoader)

    documents = loader.load()

    return documents


extracted_data = load_pdf("data/")


# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


text_chunks = text_split(extracted_data)


# download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


embeddings = download_hugging_face_embeddings()


# Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


# Creating Embeddings for Each of The Text Chunks & storing
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# If we already have an index we can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)
