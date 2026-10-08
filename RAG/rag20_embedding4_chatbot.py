# 19-2 카피
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

#3. 데이터를 벡터화한다/임베딩
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',
    model_kwargs = {
        "device" : "cpu"
    }
)
# embeddings = OpenAIEmbeddings(
#     model='text-embedding-3-small',
#     api_key=api_key,
#     base_url=base_url,
# )

#4. 벡터를 저장한다

DB_PATH = './_db/Chroma20/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma20_BAAI_bge_m3'
)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k":2})

# model
model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens = 1000,
    api_key = api_key,
    base_url = base_url
)

# prompt
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 사용자가 명시적으로 요청하는 경우에만 외부 데이터를 활용하고 컨텍스트에 없는 데이터를 활용한 경우 활용한 데이터의 범위와 함께,
'해당 정보는 컨텍스트 외부에 있는 데이터를 활용하였습니다.'라고 말씀해주세요.

컨텍스트 : {context}
질문 : {input}
답변 :
""")

# chain
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

# gradio
def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

demo = gr.ChatInterface(fn=answer_invoke, title='rag20')
demo.launch()