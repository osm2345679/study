# 17-1 카피
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

#3. 데이터를 벡터화한다/임베딩한다
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

############# 준비 완료 #############
# 이 아래가 langchain에서 제공하는 FAISS 클래스를 활용한 저장 방법

# db = FAISS.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding=embeddings
# )

DB_PATH = './_db/Faiss17/'

# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("=========================")

# 문서 저장소 id 확인
print(db.index_to_docstore_id)
# {0: '4ffd5807-a848-42fa-a3ec-f2f2fea2ee8e', 1: '8ec47e2d-b574-4405-8073-7540c7426805', ..., 17: '0eb52b61-2a6e-490b-a531-03fa97c116cd'}
print("=========================")

# 저장된 결과 확인
print(db.docstore._dict)
# '4ffd5807-a848-42fa-a3ec-f2f2fea2ee8e': Document(id='4ffd5807-a848-42fa-a3ec-f2f2fea2ee8e',
#  metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 ... 정확하다.'),
print("=========================")

# 유사도 검색
aaa = db.similarity_search("삼성전자 창업자에 대해 알려줘", k=2)
print(aaa)

# Chroma와 비교
# db = Chroma(
#     embedding_function=embeddings,
#     persist_directory=DB_PATH,
#     collection_name='croma11'
# )

# # 저장된 데이터 확인
# print("=========================")
# print(db.get())

# print("=========================")
# aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2)  # 인자 k의 디폴트는 4. 유사도가 높은 것 k개 가져오기.

# print(aaa)