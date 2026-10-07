# 15 카피
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

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = DB_PATH,
    collection_name = 'croma12'
)

################ Retrievers / 검색기 ################
retriever = vector_store.as_retriever(search_kwargs={"k":2})

print("==========================================")
################ 모델 연결 ################

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=500,
    api_key=api_key,
    base_url=base_url
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""

컨텍스트 : {context}
질문 : {input}
답변 :
""")

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)    # prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # 검색 | docu_chain

"""
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})
# input만 넣고 context를 비워도 잘 작동하는 이유 :
# create_retrieveral_chain이 인자로 들어간 retriever로부터(vector store로부터) context 알아서 가져옴

print(response)

print("============== keys() ===============")
print(response.keys())  # dict_keys(['input', 'context', 'answer'])

print("============= context ==============")
print(response['context'])
# [Document(id='1ea81773-72fb-4388-b263-e696e0a4b246',
#  metadata={'source': './_data/rag_data\\samsung_outlook.txt'},
#  page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
#  여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로,
#  회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), Document(id='b866fb5c-3577-480c-8e81-f017661eacbb',
#  metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선,
#  파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다.
#  전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')]

print("============= answer ===============")
print(response['answer'])   # 주어진 정보로는 답변할 수 없습니다.
"""

############################### gradio ###############################
############################### gradio ###############################

import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='rag15')

# Gradio 실행
demo.launch()