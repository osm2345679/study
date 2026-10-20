# tool calling : 외부 기능을 가져와서 사용!!!
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

#################################################
# 그래프 이미지 저장
#################################################
path = './_save/image/'

png_data = graph.get_graph().draw_mermaid_png()

with open(path + "lg07_ToolNode.png", 'wb') as file:
    file.write(png_data)

# with는 close와 함께. 그냥 아래와 같음.
# file = open(path + "lg07_ToolNode.png", 'wb')
# file.write(png_data)
# file.close()