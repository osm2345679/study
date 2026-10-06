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
# 3. 데이터를 벡터화한다
# 4. 데이터를 벡터 DB에 저장한다

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

DB_PATH = './_db/Chroma/'

db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma11'
)

# 저장된 데이터 확인
print("=========================")
print(db.get())

print("=========================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2)  # 인자 k의 디폴트는 4. 유사도가 높은 것 k개 가져오기.

print(aaa)