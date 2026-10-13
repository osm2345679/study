# 11-1 카피
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma 
import os

# pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

# 할 일
# 1. 데이터를 불러온다
# 2. 데이터를 자른다/청킹한다
# 3. 데이터를 벡터화한다/임베딩한다
# 4. 데이터를 벡터 DB에 저장한다

#1. 데이터를 불러온다
path = './_data/rag_data/'
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')

#2. 문서(데이터)를 자른다 / 청킹한다
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""] # 통상 디폴트
)

split_doc1 = loader1.load_and_split(text_splitter) # 청크 300, 오버랩 100
split_doc2 = loader1.load_and_split(text_splitter) # 청크 300, 오버랩 100

# 문서 개수 확인
# print(split_doc1)
print(len(split_doc1), len(split_doc2))  # 9 9

#3. 데이터를 벡터화한다/임베딩한다
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

#4. 데이터를 벡터 DB에 저장한다
############### FAISS ###############
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hi")))  # faiss_index = faiss.IndexFlatL2(1536)

# 벡터 store의 벡터 차원 수(임베딩 차원 수)
print(faiss_index.d)    # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal)    # 0

############# 준비 완료 #############
# 이 아래가 langchain에서 제공하는 FAISS 클래스를 활용한 저장 방법

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings
)

DB_PATH = './_db/Faiss17'
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)

# Chroma와 비교
# db = Chroma.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding=embeddings,
#     persist_directory=DB_PATH,
#     collection_name='croma11'
# )