#load pdf raw data
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load the file
DATA_PATH="data/"
def load_pdf_file(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents=load_pdf_file(DATA_PATH)
print("length of pdf pages ", len(documents))


#create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("length of text chunks ", len(text_chunks))
#create vector embedding

def get_embedding_model():
    embeding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeding_model
embeding_model = get_embedding_model()
print("embedding model loaded")


#store embeding in FAIS

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks,embeding_model)
db.save_local(DB_FAISS_PATH)
