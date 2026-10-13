# 20-2 카피
# LCEL = Langchain Expression Language
# chian = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# 06에서 카피함
api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     model = 'text-embedding-3-small',
#     api_key=api_key,
#     base_url=base_url
# )

# HuggingFaceEmbeddings만 install 하면 아래처럼 에러남.
# ImportError: Could not import sentence_transformers python package. Please install it with `pip install sentence-transformers`.
# sentence-transformers도 install
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name = "BAAI/bge-m3",
#     model_kwargs = {
#         "device" : "cpu",
#         # "local_files_only" : True
#     }
# )

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs = {
        "device" : "cpu",
        # "local_files_only" : True # 한 번 실행해서 로컬에 다운받고 나면 이 옵션 키면 좋음.
    }
)

vector = embeddings.embed_query(prompt)
print(vector)
print("================================")
print("임베딩 벡터의 차원 : ", len(vector)) # 임베딩 벡터의 차원 :  1024