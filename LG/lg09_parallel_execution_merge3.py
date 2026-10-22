# tool calling : 외부 기능을 가져와서 사용!!!
# 9-2 카피
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    max_tokens=4000,
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

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 메모리 추가 ㄱ
from langgraph.checkpoint.memory import InMemorySaver

#1. state 정의
class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str

#2. node 정의
# 검색 에이전트
def search_agent(state: State):
    print("검색 에이전트 실행")
    question = state["messages"][-1].content

    result = web_search.invoke({
        "query" : question
    })
    return {"search_result" : str(result)}

def analysis_agent(state: State):
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response = llm.invoke(
        f"다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}"
    )
    return {"analysis_result" : response.content}

def merge_agent(state: State):  # 결과 통합 에이전트!
    # result = (
    #     state["search_result"] + "\n\n" +
    #     "================================" + "\n" +
    #     state["analysis_result"]
    # )
    # return {"messages" : [("assistant", result)]}

    print("결과 통합 에이전트 실행")

    prompt = f'''
    다음 두 결과를 종합하여 최종 답변을 작성해줘.

    [웹 검색 결과]
    {state["search_result"]}

    [AI 분석 결과]
    {state["analysis_result"]}

    검색 결과를 우선적으로 참고하고,
    최신 정보와 기존 분석을 구분해서 설명해줘.
    '''

    response = llm.invoke(prompt)

    return {"messages" : [response]}

#3. graph 구성
builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

######### 병렬 실행 #########
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

######### 병합 #########
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

#4. 컴파일
graph = builder.compile()

#5. 실행
query = "엔비디아의 미래 전망은?"

response = graph.invoke({
    "messages" : [("user", query)]
})

print("Assistant :", response["messages"][-1].content)

########### 그림 그리기 ###########
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# Results
# 분석 에이전트 실행
# 검색 에이전트 실행
# Assistant : {'query': '엔비디아의 미래 전망은?', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://www.litefinance.org/ko/blog/analysts-opinions/nvda-juga-jeonmang', 'title': '2026년, 2027년, 2028~2030년, 2040년 및 향후 엔비디아(NVDA) 주식 예측 및 주가 전망 | 라이트파이낸스', 'content': '엔비디아의 장기 주가 전망은 분석 플랫폼별 가치평가 모델과 회사의 미래에 대한 가정이 서로 다름에 따라 전망별로 큰 편차를 보이고 있습니다. 장기적인 주가 흐름은 AI 도입 속도, 컴퓨팅 성능 수요, 데이터 센터 시장 확대, 반도체 산업에서 엔비디아가 선도적 지위를 유지할 수 있는지에 따라 좌우될 것으로 보입니다.\n\n스탁스캔에 따르면 엔비디아는 장기 상승 추세를 이어갈 것으로 예상됩니다. 평균 주가는 2035년 $28,690.77에 이를 것으로 전망되며, 2040년에는 $37,489.98, 2050년에는 $45,762.65까지 상승할 것으로 예상됩니다.\n\n코인프라이스포캐스트는 이보다 보수적인 전망을 제시하고 있습니다. 엔비디아 주가는 2033년까지 $567.00, 2035년에는 $619.00까지 상승할 것으로 보입니다. [...] 2026년 전망에 따르면 엔비디아 주가는 $164.75~$284.00 범위에서 거래될 것으로 예상됩니다. 일부 애널리스트들은 완만한 조정 후 회복할 것으로 보는 반면, 일부 애널리스트들은 꾸준한 상승세가 이어질 것으로 전망하고 있습니다. 변동성은 연중 높게 유지될 것으로 예상됩니다.\n\n장기 전망에 따르면 엔비디아 주가는 5년 이내에 $567를 넘어설 가능성이 있으며, 2035년에는 $28,690.77에 도달할 것으로 예상됩니다. 이처럼 전망치의 차이가 큰 것은 엔비디아의 성장 가능성과 인공지능 시장 전망에 대한 애널리스트들의 견해가 크게 엇갈리고 있기 때문입니다.\n\n2030년 컨센서스에 따르면 엔비디아는 $493.39~$982.89 범위에서 거래될 것으로 예상됩니다. 대부분의 전문가들은 장기 상승 추세가 이어질 것으로 보고 있지만, 일시적인 조정이나 횡보 국면이 나타날 가능성도 있습니다. [...] ## 엔비디아 주가 전망 관련 자주 묻는 질문\n\n17.10.2026 기준 엔비디아의 주가는 $214.11입니다.\n\n대부분의 전망에서는 엔비디아의 장기 상승 추세가 유지될 것으로 보고 있지만, 조정과 높은 변동성이 나타날 가능성도 있습니다. 장기 투자자라면 기존 포지션을 유지하거나 추가 매수하는 전략을 고려할 수 있습니다.\n\n애널리스트들은 향후 12개월 동안 엔비디아 주가가 $164.75~$284.00 범위에서 움직일 것으로 예상하고 있습니다. 전망은 엇갈리는 모습을 보이고 있습니다. 일부 전문가들은 완만한 등락을 예상하는 반면, 일부 전문가들은 점진적인 상승을 전망하고 있습니다.\n\n네. 대부분의 전문가들은 꾸준한 상승을 전망하고 있습니다. 엔비디아는 AI와 칩 산업에서 지배적인 위치를 차지하고 있으며, 높은 투자 매력도와 긍정적인 주가 모멘텀을 유지하고 있습니다.', 'score': 0.8891543, 'raw_content': None, 'id': '5e5d4e-00'}, {'url': 'https://www.youtube.com/watch?v=MwGXc35X0BU', 'title': 'NVIDIA 주가 전망: 2030년까지 심층 분석(4조 달러의 질문)', 'content': '제공된 문서는 엔비디아(NVIDIA) 주가 전망에 대한 현재부터 2030년까지의 다차원적 종합 평가를 제공합니다. 이 분석은 엔비디아가 AI 혁명의 핵심 인프라 제공자로 자리매김한 새로운 위상을 서론에서 강조합니다. 보고서는 2025년 7월 기준의 재무적 위상을 깊이 있게 다루며, P/E 비율, 총마진율(약 70%), 순이익률(50% 이상) 등 핵심 지표의 강력함을 제시합니다. 또한, 주가 전망을 상승론(Bull Case)과 하락론(Bear Case)이라는 양면적인 논리 구조로 비교 분석하며, 장기적인 성장률 둔화 리스크와 높은 밸류에이션에 대한 우려도 함께 논합니다. 결론적으로, 알고리즘 예측 모델이 제시하는 2030년 1,000달러 이상의 낙관적 전망과 함께 경쟁 심화 및 규제 리스크에 대한 균형 잡힌 시각을 투자자들에게 제시합니다. [...] ### Transcript\n[0:00] AI 혁명에 가장 중심에서 있는 기업 바로 엔비디아죠. 지금 엔비디아의 미래를 두고 정말 말이 많습니다.\n[0:08] 한쪽에서는 이건 시작일 뿐이다 하고 다른 쪽에선 아니다. 곧 조정이 올 거다라고 하죠. 오늘 저희가이 팽팽한\n[0:16] 논쟁의 한 가운데로 들어가서 그 핵심을 제대로 한번 펼쳐 보겠습니다.\n[0:22] 자, 시작부터 한번 보세요. 171% 이게 무슨 숫자일까요?\n[0:28] 2024년 단 한 해 동안 엔비디아 주가가 오른 수치입니다. 정말 믿기 힘든 숫자죠. 바로이 엄청난 상승세가\n[0:36] 오늘 우리가 엔비디아이기를 해야만 하는 이유입니다. 자, 그럼 엔비디아의 미래를 이야기하기 전에 우리가 먼저 짚고 넘어가야 할게\n[0:44] 있습니다. 바로 지금 엔비디아는 어떤 회사인가 하는 거죠.\n[0:49] 괜히 AI 시대의 황제라고 불리는게 아니거든요. 그 압도적인 지배력이 어디서 나오는지부터 한번 살펴보겠습니다. [...] [1:55] 모든 비용 다 떼고 순수하게 남는 돈도 50%가 넘어요. 이건 뭐 업계에서는 거의 사기 수준이죠.이\n[2:03] 숫자들 자체가 엔비디아의 기술이 얼마나 독보적인지 그리고 가격을 내가 정한다고 말할 수 있는 힘이 얼마나\n[2:11] 센지를 그냥 그대로 보여 주는 겁니다. 자, 지금까지 Nv디아의 현재 모습을 봤는데요. 그럼 이제\n[2:18] 본격적으로 오늘 이야기의 핵심으로 들어가 보죠. 바로 NBDI의 미래를 보는 두 개의 시선, 즉 끝없는\n[2:25] 성장을 외치는 낙관론 불케이스와 아니다. 조심해야 한다고 말하는 신중론 베어케이스의 팽팽한\n[2:32] 대결입니다. 과연 엔비디아의 미래는이 둘 중 어느 쪽을 향하게 될까요?\n[2:39] 지금부터이 두 주장이 각각 어떤 근거를 가지고 있는지 하나하나 정면으로 부딪쳐 보겠습니다. 여기에이 논증의 핵심이 딱 정리되어 있습니다.', 'score': 0.8649622, 'raw_content': None, 'id': 'c35dab-01'}, {'url': 'https://www.bbc.com/korean/articles/c1j8j51yzk6o', 'title': '엔비디아, 실적·전망 모두 예상치 상회 … 주가 급등 - BBC News 코리아', 'content': '# 엔비디아, 실적·전망 모두 예상치 상회… 주가 급등\n\n젠슨 황 엔비디아 CEO\n\n사진 출처, Reuters\n\n반도체 대기업 엔비디아가 월가 예상치를 웃도는 매출과 향후 실적 전망을 기록하며, 시장을 불안하게 했던 막대한 AI 지출에 대한 투자자들의 걱정을 누그러뜨렸다.\n\n엔비디아 측은 지난 19일(현지시간) 분기 실적 발표에서 올해 3분기(8~10월) 매출이 570억달러(약 83조원)로 전년 동기보다 62% 증가했다고 밝혔다. 이는 AI 데이터 센터용 반도체 수요 증가에 힘입은 결과다. 해당 부문의 매출은 66% 증가하여 510억달러를 넘어섰다.\n\n4분기 매출 역시 650억달러로 예상했는데, 이 또한 예상치를 웃도는 값으로, 이러한 소식에 힘입어 엔비디아 주가는 시간 외 거래에서 4% 이상 급등했다\n\n세계 1위의 시가총액을 자랑하는 엔비디아의 주가는 AI 붐의 선행지표로 여겨진다. 엔비디아의 실적은 시장 심리에도 영향을 미칠 수도 있다. [...] End of TOP 뉴스\n\n\'AI 거품론\'으로 인해 AI 투자 수익률에 대한 의문이 높아지며 지난 19일 까지 미국 S&P 500 지수는 4일 연속 하락했다. S&P 500 지수는 이번 달 들어 3% 가까이 하락한 상태다.\n\n한편 엔비디아의 이번 실적에 대한 시장의 기대치는 높았다.\n\nEnd of content\n\n미국 투자 자문 서비스 \'LPL 파이낸셜\'의 애덤 턴퀴스트 수석전략가는 엔비디아가 시장의 기대치를 뛰어넘는지 여부가 중요한 게 아니라 "얼마나 크게 넘어섰는가"가 관건이었다고 설명했다.\n\n영국의 \'하그리브스 랜즈다운\'의 매트 브릿즈먼 선임애널리스트 또한 "AI 기업들의 가치평가가 연일 뉴스에 오르내리고 있는 가운데 엔비디아는 계속 흔들림 없이 자신들의 사업을 이어가고 있다"고 설명했다.\n\n이어 AI 산업의 일부 분야는 "잠시 숨 고르기가 필요했지만, 엔비디아는 여기에 속하지 않는다"고 덧붙였다. [...] 과거 황 CEO는 내년까지 AI 반도체 주문 규모가 5000억달러에 달할 것으로 설명한 바 있다. 투자자들은 회사가 언제 이러한 매출을 실현할 수 있을지, 또 어떻게 이러한 주문을 실제 이행할 수 있을지에 관한 세부 정보를 기다리고 있었다.\n\n엔비디아의 콜레트 크레스 최고재무책임자(CFO)는 애널리스트들에게 이미 언급한 5000억달러 규모 주문에 더해 "아마도" 추가 주문을 더 받을 것 같다고 귀띔했다.\n\n동시에 자사의 대중 반도체 수출을 가로막는 규제 제한에 실망감을 드러내며, 미국은 중국을 포함하여 "모든 개발자 시장의 지지를 얻어야 한다"고 강조했다.\n\n크레스 CFO는 엔비디아는 미국과 중국 정부와 "지속적으로 협력하며 최선을 다하고 있다"고 전했다.', 'score': 0.7208996, 'raw_content': None, 'id': '1f7ed4-02'}], 'response_time': 0.0, 'request_id': '0dddf26f-d046-4dbe-9509-d0d01bbb9e45'}

# ================================