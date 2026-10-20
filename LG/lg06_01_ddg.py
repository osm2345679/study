# tool calling : 외부 기능을 가져와서 사용!!!
# 05 카피
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
# # pip install langchain-tavily
# from langchain_tavily import TavilySearch

# # 웹 검색 도구 생성
# web_search = TavilySearch(
#     max_results=3
# )

# pip install ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun(
    max_results=5
)

# results = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })

# print(results)

tools = [web_search]

llm_with_tools = llm.bind_tools(tools=tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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
graph = graph_builder.compile()

#6. 실행
query = "삼성전자의 오늘 주가를 알려줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

print("AI : ", response["messages"][-1].content)