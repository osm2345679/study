# tool calling : 외부 기능을 가져와서 사용!!!
# 8-1 카피
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

############# 0. Tool 정의 ##############
# pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=3
)

# results = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })

# print(results)
# {'query': '삼성전자의 오늘 주가 알려줘', 'follow_up_questions': None,
# 'answer': None, 'images': [], 'results': [{'url': 'https://kr.investing.com/equities/samsung-electronics-co-ltd',
#  'title': 'Samsung Electronics Co Ltd 오늘의 주가 | 005930 실시간 티커 - Investing.com', 'content': '### 주요 주주...',
#  'score': 0.7520312, 'raw_content': None, 'id': '463907-02'}], 'response_time': 0.0, 'request_id': '3f1a6e58-e16a-4c57-8c22-732bddf1aad3'}

# print(results['results'][0]['content'])
# 주요 주주 ... 적정가치

tools = [web_search]

llm_with_tools = llm.bind_tools(tools=tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 메모리 추가 ㄱ
from langgraph.checkpoint.memory import InMemorySaver

#1. state 정의
class State(TypedDict):
    messages : Annotated[list, add_messages]

#2. node 정의
def chatbot(state : State):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}

#3. graph 구성, node 추가
# graph 생생성
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools)
# graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tools", ToolNode(tools))    # 위 두줄이랑 같음

#4. edge 추가
graph_builder.add_edge(START, "chatbot")

# graph_builder.add_edge(chatbot, "tools")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition # tool_calls가 있는지
    # tool_calls가 있으면 -> 'tools' 반환
    # tool_calls가 없으면 -> END 반환
    # tools_condition에 END가 포함되어 아래에 END와 edge 연결해줄 필요 없음
)

graph_builder.add_edge("tools", "chatbot")

#5. 컴파일
# 메모리 추가 ㄴ
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

#6. 실행
# 메모리 추가 ㄷ
config = {"configurable" : {"thread_id" : "usesr01"}}

# 첫번째 질문
response = graph.invoke(
    {"messages" : [("user", "내 이름은 홍길동이야.")]},
    config=config)

print(response["messages"][-1].content)

print("================ 두 번째 질문 ================")
# 두번째 질문
response = graph.invoke(
    {"messages" : [("user", "내 이름이 뭐라고 했지?")]},
    config=config)
print(response["messages"][-1].content)

# Results
# 반가워요, 홍길동님! 무엇을 도와드릴까요?
# ================ 두 번째 질문 ================
# 홍길동이라고 하셨어요.