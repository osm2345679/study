# 23-1 카피
from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

############### BM25 검색 ################
# BM25는 키워드 기반 검색. Best Match
# from langchain_community.retrievers import BM25Retriever

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3    # best 3개 문장

# bm25_docs = bm25_retriever.invoke(query)

# print("==================== BM25 검색 결과 ====================")
# for doc in bm25_docs:
#     print(doc.page_content)
# ==================== BM25 검색 결과 ====================
# 삼성전자는 AI 반도체 기술을 연구한다.
# 반도체는 컴퓨터의 핵심 부품이다.
# 테슬라는 전기자동차를 제조한다.


###### 벡터 검색 ######

# 22-2 카피
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

#3. 데이터를 벡터화한다/임베딩
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

#4. 벡터를 저장한다

DB_PATH = './_db/Chroma23/'

vector_store = Chroma.from_documents(
    embedding = embeddings,
    documents = texts,
    persist_directory = DB_PATH,
    collection_name = 'croma23_2'
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)
vector_docs = vector_retriever.invoke(query)

print("==================== vectorDB 검색 결과 ====================")
for doc in vector_docs:
    print(doc.page_content)
# ==================== vectorDB 검색 결과 ====================
# 삼성전자는 반도체를 생산하는 기업이다.
# 삼성전자는 AI 반도체 기술을 연구한다.
# 삼성전자는 갤럭시 스마트폰을 판매한다.