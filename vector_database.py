from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

#Step 1: Upload & Load raw PDF(s)

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


file_path = 'universal_declaration_of_human_rights.pdf'
documents = load_pdf(file_path)
#print("PDF pages: ",len(documents))

#Step 2: Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
#print("Chunks count: ", len(text_chunks))


load_dotenv()
#Step 3: Setup Embeddings Model (Use lightweight Ollama embedding model by default)
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
def get_embedding_model(model_name: str):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings

#Step 4: Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(OLLAMA_EMBED_MODEL))
faiss_db.save_local(FAISS_DB_PATH)
