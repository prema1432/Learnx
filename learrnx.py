import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title('Hey there! Welcome to Learnx')

st.sidebar.title('Tarin Your Data')


def user_input(user_question):
    st.write("Reply : ", user_question)


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
                elif file_type == "File Upload":
                    if not upload_file:
                        st.warning('Please upload a file')
                else:
                    st.warning('Invalid File type')

                st.success('Your Data Training is completed')


if __name__ == '__main__':
    main()
