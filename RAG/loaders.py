from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader

loader=TextLoader("./data/text_files/python_intro.txt")
document=loader.load()
print(document[0].page_content)
print(document[0].metadata)