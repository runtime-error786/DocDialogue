import streamlit as st
from src.QA.QA import setup_qa_system, answer_query

st.title("Document-Based Question Answering")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # Setup QA system with the uploaded document
    qa_chain = setup_qa_system(uploaded_file)

    st.write("Document uploaded successfully. You can now ask questions about the document.")

    query = st.text_input("Ask a question about the document:")
    
    if query:
        response = answer_query(qa_chain, query)
        st.write(f"Answer: {response}")
