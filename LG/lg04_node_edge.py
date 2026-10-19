# 04 카피
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

#1. state 정의
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int

# state ={
#     "messages" : [],
#     'aaa' : 'abc'
# }

#2. node 정의
def prepare(state : State) :
    return {"messages" : [AIMessage(content="질문을 접수했습니다.")]}

def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}

def finish(state : State) :
    return {"messages" : [AIMessage(content="답변 처리가 완료되었습니다.")]}

#3. graph 구성
graph_01 = StateGraph(State)
graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)

#4. 엣지 추가
graph_01.add_edge(START, "prepare")
graph_01.add_edge("prepare", "chatbot")
graph_01.add_edge("chatbot", "finish")
graph_01.add_edge("finish", END)

#5. graph 컴파일
graph = graph_01.compile()

#6. graph 실행
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

#7. 결과 출력
for i, msg in enumerate(response['messages']):
    print(f"[{i}] {msg.type} :{msg.content}")

# [0] human :삼성전자의 주가를 예측해줘
# [1] ai :질문을 접수했습니다.
# [2] ai :삼성전자 주가를 **정확히 예측하는 것은 ... 반드시 확인하세요.
# [3] ai :답변 처리가 완료되었습니다.