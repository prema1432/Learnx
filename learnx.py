import streamlit as st

from dotenv import load_dotenv
import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title('Hey there! Welcome to Learnx')

st.sidebar.title('Tarin Your Data')

def load_prompt():
    prompt = """
    You need to anser the questionn in the senntence as same as in the conntent..
    
    Give n newlow is the context and question of the user.
    context = {context}
    question = {question}
    if the answer is not in the context then you need to answer the question in the sentence as same as in the conntent.. is not in the context, you can say "I don't know"
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # retrive_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever(),
    #                                             return_source_documents=True, verbose=False)
    # result = retrive_chain.invoke(user_question)


    prompt = load_prompt()

    rag_chain =(
        {"context":vector_db.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(user_question)

    st.write("Reply : ", result)


def main():
    st.header('Gen AI RAG')

    user_question = st.text_input('Enter your question')
    if user_question:

        user_input(user_question)

    with st.sidebar:
        additional_text = st.text_input('Enter additional text', placeholder='Enter additional text')
        file_type = st.selectbox("Select file type", ("Plain Text", "File Upload"))
        upload_file = st.file_uploader("Upload a file")

        if st.button('Submit & Process ....', key="process_button"):
            with st.spinner('Processing...'):
                import tempfile

                if file_type == "Plain Text":
                    if not additional_text:
                        st.warning('Please enter additional text')

                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                        temp_file.write(additional_text)
                        temp_file_name = temp_file.name
                    with open(temp_file_name) as file:
                        content = file.read()
                    loader = TextLoader(temp_file_name)
                    documents = loader.load()
                    # st.write(documents)




                elif file_type == "File Upload":
                    if not upload_file:
                        st.warning('Please upload a file')
                else:
                    st.warning('Invalid File type')

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
                docs = text_splitter.split_documents(documents)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_db = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="chroma_db")
                st.success('Your Data Training is completed')


if __name__ == '__main__':
    main()
