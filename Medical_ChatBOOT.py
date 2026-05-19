import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()
HF_Token = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

CUSTOM_PROMPT_TEMPLATE = """
use the pieces of information provided in the context to answer the user's question.
if you don't know the answer, just say that you don't know the answer, don't try to make an answer.
don't provide anything out of the given context.
Context: {context}
Question: {question}

start the answer directly. No small talk please.
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# Create QA chain
def load_llm(HUGGINGFACE_REPO_ID, HF_Token):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_Token, "max_length": 512}
    )
    return llm

def main():
    st.title("Medical ChatBOOT")
    prompt = st.chat_input("You:")

    if 'message' not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        st.chat_message(message["role"]).markdown(message["content"])

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.message.append({"role": "user", "content": prompt})

        llm = load_llm(HUGGINGFACE_REPO_ID, HF_Token)

        boot_response = "Hi, I am a Medical ChatBOOT. I am here to help you with your medical queries."
        st.chat_message("bot").markdown(boot_response)
        st.session_state.message.append({"role": "bot", "content": boot_response})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vectorstore is not loaded")
            else:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt()}
                )
                boot_response = qa_chain.invoke({"query": prompt})

                result = boot_response["result"]
                source_documents = boot_response["source_documents"]

                result_to_show = result + "\n\n" + "\n\n".join([doc.page_content for doc in source_documents])

                st.chat_message("bot").markdown(result_to_show)
                st.session_state.message.append({"role": "bot", "content": result_to_show})
        except Exception as e:
            st.error(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
