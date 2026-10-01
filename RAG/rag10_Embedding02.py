# 10-1 카피
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    dimensions=5
)

vector = embeddings.embed_query(prompt)
print(vector)
print("================================")
print("임베딩 벡터의 차원 : ", len(vector))

# [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875]
# ================================
# 임베딩 벡터의 차원 :  5