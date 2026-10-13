# 22-2 카피
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

################3 Retriever #######################
# retriever = vector_store.as_retriever(search_kwargs={"k":2})
retriever = vector_store.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={
        "k":2,
        "score_threshold":0.7
    }
)

# model
model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens = 1000,
    api_key = api_key,
    base_url = base_url
)

# prompt
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 강력한 AI 챗봇입니다.
    
    다음 규칙에 따라 답변하세요.
    1. 이전 대화에서 확인할 수 있는 정보는 이전 대화를 참고하여 답변하세요.
    2. 일반적인 인사나 일상 대화는 자연스럽게 답변하세요.
    3. 문서에 관한 질문은 제공된 컨텍스트를 근거로 답변하세요.
    4. 문서에 관한 질문인데 컨텍스트에 관련 정보가 없다면
    "주어진 정보로는 답변할 수 없습니다"라고 답변하세요.

    컨텍스트 : {context}
    답변 :
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
])

# chain
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

# gradio
def answer_invoke(message, history):
    chat_history = []

    for item in history[-10:]:  # 최근 10개의 메세지만 전달
        content = item['content']

        if item['role'] == "user":
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == "assistant":
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
    })

    # retriever = vector_store.as_retriever(search_kwargs={"k":2}) 에서 설정한 2개의 문서 출력해보기
    for i, doc in enumerate(response['context'], start=1):
        print(f"\n검색 문서 {i}")
        print("출처 :", doc.metadata.get("source"))
        print("내용 :", doc.page_content)
        print("=================================")

    return response['answer']

demo = gr.ChatInterface(fn=answer_invoke, title='rag22')
demo.launch()

'''
1. 안녕
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
]

2. 이 프로젝트는 rag21이야.
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
    {"role" : "user", "content" : "이 프로젝트는 rag21이야"}
    {"role" : "assistant", "content" : "이 프로젝트는 rag21이군요"}
]

3. 이 프로젝트는 뭐게?
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
    {"role" : "user", "content" : "이 프로젝트는 rag21이야"}
    {"role" : "assistant", "content" : "이 프로젝트는 rag21이군요"}
    {"role" : "user", "content" : "이 프로젝트는 뭐게?"}
    {"role" : "assistant", "content" : "이 프로젝트는 rag21입니다"}
]
'''