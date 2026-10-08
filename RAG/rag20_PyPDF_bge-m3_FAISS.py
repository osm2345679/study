from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

#1. 데이터를 불러온다
path = './_data/'
pdf_loader = PyPDFLoader(path + '1706.03762v7.pdf')
pdf_docs = pdf_loader.load()

# print(type(pdf_docs))   # <class 'list'>
# print(len(pdf_docs))    # 15
# print(pdf_docs)
# print(pdf_docs[0])


#2. 데이터를 자른다/청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

split_docs = text_splitter.split_documents(pdf_docs)

#3. 데이터를 벡터화한다/임베딩
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',
    model_kwargs = {
        "device" : "cpu",
        "local_files_only" : True
    }
)

#4. 벡터를 저장한다

db = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)

DB_PATH = './_db/FAISS20_bge-m3/'

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index20_bge-m3'
)