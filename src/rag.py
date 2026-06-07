from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = TextLoader(
    file_path='../data/sample_text.txt'
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
doc_chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
    api_key=api_key
)

vector_store = FAISS.from_documents(doc_chunks, embeddings)

retriever = vector_store.as_retriever(
	 search_type="similarity",
	 search_kwargs={"k": 5}
)
retrieved_documents = retriever.invoke("Why do compilers use abstract syntax trees?")

for i in range(0, len(retrieved_documents)):
	print(retrieved_documents[i].page_content)