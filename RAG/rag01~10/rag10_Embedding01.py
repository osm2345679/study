# 8-2 카피
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

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

vector = embeddings.embed_query(prompt)
print(vector)
print("================================")
print("임베딩 벡터의 차원 : ", len(vector))

# [0.038177490234375, -0.022735595703125, ..., -0.0236053466796875, 0.0100631713867187
# ================================
# 임베딩 벡터의 차원 :  1536

# model = 'text-embedding-3-large'로 바꾸면
# ================================
# 임베딩 벡터의 차원 :  3072