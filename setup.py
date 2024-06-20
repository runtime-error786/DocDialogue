from setuptools import find_packages,setup

setup(
    name="QA",
    version='0.0.1',
    author='Mustafa Rizwan',
    author_email='mustafa782a@gmail.com',
    install_requires=['openai','langchain','streamlit','python-dotenv','PyPDF2','chromadb','langchain_community','langchain_text_splitters','langchain_chroma'],
    packages=find_packages()
)