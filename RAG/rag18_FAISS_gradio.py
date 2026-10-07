from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import gradio as gr
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url
)

DB_PATH = './_db/Faiss17/'

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

# 모델 연결

model = ChatOpenAI(
    model='gpt-5.6-terra',
    temperature=0,
    max_tokens=500,
    api_key=api_key,
    base_url=base_url
)

# 프롬프트
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해주세요.

컨텍스트 : {context}
질문 : {input}
답변 :
""")

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

# gradio
def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

demo = gr.ChatInterface(fn=answer_invoke, title='rag18')
demo.launch()