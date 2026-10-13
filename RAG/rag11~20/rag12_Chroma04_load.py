# 11-3 카피
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma 
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key = api_key,
    base_url = base_url
)

DB_PATH = './_db/Chroma12/'

# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_PATH,
#     collection_name='croma12'
# )
vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = DB_PATH,
    collection_name = 'croma12'
)
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")  # 벡터 저장소에 저장된 문서 수 :118. 2번 실행해서 2배 됨. 원래 59.

query = " 삼성전자의 창업자는 누구인가요?"
results = vector_store.similarity_search(query)

print(f"검색 결과의 길이 : {len(results)}") # 검색 결과의 길이 : 4

################ Retrievers / 검색기 ################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)    # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000019EFF8A6C10> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")  # 검색된 관련 문서 수 : 2
print(f"첫 번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}")   # 삼성전자 사업 전망 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사