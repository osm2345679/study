# 11-1 카피
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma 
import os

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
from glob import glob

path = './_data/rag_data/'

#폴더에서 텍스트 문서 가져오기
txt_files = glob(os.path.join(path, '*.txt'))

print(txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']

#리스트 형태의 문서 경로들에서 for문으로 데이터 가져오기
data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    # data.append(loader)
    data += loader.load()

print("======================================")
print(len(data))    # 3
print(data[0])

char_count = [len(doc.page_content) for doc in data]
print(char_count)   # [8158, 2049, 1898]


#2. 문서(데이터)를 자른다 / 청킹한다
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""] # 통상 디폴트
)

texts = text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts)) # 생성된 텍스트 청크수 : 59
print("각 청크의 길이 :", list(len(text.page_content) for text in texts))
# 각 청크의 길이 : [259, 154, 150, 282, 128, 276, ... , 170, 187, 249]

print("첫 번째 청크의 내용 :", texts[0].page_content)
# 첫 번째 청크의 내용 : page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산

# 작성 목적
# 이 문서는 2026년 10월 초 공개된 국내 인공지능 관련 최신 보도를 바탕으로 LangChain의 문서 로딩, 텍스트 분할,
#  임베딩, 벡터 데이터베이스 저장, 검색 및 RAG 실습에 활용할 수 있도록 재구성한 학습용 텍스트이다. 특정 언론사의
#  기사 원문을 복제하지 않고 공개된 사실과 기술적 배경을 중심으로 내용을 확장해 작성하였다.

# 1. 한국의 AI for All 프로젝트'
print("첫 번째 청크의 길이 :", len(texts[0].page_content)) # 첫 번째 청크의 길이 : 259


#3. 데이터를 벡터화한다/임베딩한다
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print(vector)
print(len(vector))  # 1536. 임베딩 완.

DB_PATH = './_db/Chroma12/'

#4. 데이터를 벡터 DB에 저장한다
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma12'
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