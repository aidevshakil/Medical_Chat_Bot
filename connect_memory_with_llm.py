import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Setup LLM (Mistral with HuggingFace)
hf_token = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": 512}
    )
    return llm

# GALE ENCYCLOPEDIA of MEDICINE Custom Template
CUSTOM_PROMPT_TEMPLATE = """
use the pieces  of information provide in the context to answer user's question .
if you dont know thw answer, just say that you dont know the answer, dont try to make a answer.
dont provide anythins out of the given context 
Context:{context}
Question:{question}


start the answer directly . No small Talk please 

"""

# Initialize Prompt Template
def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


#create qu chain
# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt()}
)


# invorke with chat method
# Example usage
user_query = input("Enter your query: ")
response = qa_chain.invoke({"query": user_query})
print({"RESULT":response["result"]})
print({"SOURCE_DOCUMENTS":response["source_documents"]})