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

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 {how} 설명해주세요.")

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url
)

chain = prompt | model

input = {"topic" : "RAG", "how" : "초등학생도 이해하기 쉽게"}

response = chain.invoke(input)
print(response.content)
'''
RAG는 **AI가 대답하기 전에 필요한 정보를 찾아보고, 그 정보를 바탕으로 답하는 방법**이에요.

영어로는 **Retrieval-Augmented Generation**이라고 해요.

- **Retrieval(찾기)**: 필요한 책, 문서, 인터넷 자료 등을 찾아요.
- **Augmented(도움받기)**: 찾은 자료를 AI에게 보여줘요.
- **Generation(답 만들기)**: AI가 자료를 읽고 알기 쉽게 답해요.

### 쉽게 비유하면

AI가 숙제를 하는 학생이라고 생각해 볼게요.

보통 AI는 자기 머릿속에 이미 배운 내용으로 답해요.  
그런데 모르는 내용이나 최신 정보는 틀릴 수 있어요.

RAG를 쓰면 AI는 이렇게 해요.

1. 질문을 받아요.  
   - “우리 학교 급식 메뉴가 뭐야?”

2. 급식표를 찾아봐요.  
   - 학교 홈페이지나 급식표 문서에서 찾아요.

3. 찾은 내용을 보고 답해요.  
   - “오늘 급식은 카레밥과 우유예요.”

즉, **시험 보기 전에 교과서나 사전을 찾아보고 답하는 똑똑한 방법**이에요.

### 왜 좋을까요?

- 최신 정보를 알려줄 수 있어요.
- 회사 문서, 학교 자료처럼 특별한 자료를 이용할 수 있어요.
- AI가 엉뚱하게 지어내는 답을 줄이는 데 도움이 돼요.
- “어디 자료를 보고 답했는지” 함께 보여줄 수도 있어요.

### 하지만 조심할 점도 있어요

RAG도 완벽하지 않아요.

- 찾아온 자료가 틀리면 답도 틀릴 수 있어요.
- 중요한 내용은 사람도 한 번 확인하는 것이 좋아요.
- AI가 자료를 잘못 이해할 수도 있어요.

한 줄로 말하면:

> **RAG는 AI가 답하기 전에 자료를 찾아보고, 그 자료를 이용해 답하는 기술이에요.**
'''

input = {"topic" : "langchain", "how" : "초등학생도 이해하기 쉽게"}

response = chain.invoke(input)
print(response.content)
'''
LangChain은 **AI를 더 똑똑하게 일을 시키기 위한 도구 상자**예요.

예를 들어 ChatGPT 같은 AI는 질문에 대답을 잘하지만, 혼자서는 보통 이런 일을 하기 어려워요.

- 인터넷에서 최신 정보 찾기
- PDF, 엑셀, 회사 문서 읽기
- 계산기 사용하기
- 데이터베이스에서 정보 찾기
- 여러 단계를 차례대로 처리하기

LangChain은 AI에게 이런 도구들을 연결해 주는 역할을 합니다.

### 쉽게 비유하면

AI를 **똑똑한 로봇**이라고 생각해 볼게요.

로봇이 “내일 비 올까?”라는 질문을 받았을 때:

- 그냥 AI만 있으면: 예전에 배운 내용으로 추측할 수 있어요.
- LangChain이 있으면:
  1. 날씨 사이트에 가서
  2. 최신 날씨 정보를 찾고
  3. 중요한 내용을 읽고
  4. “내일 오후에 비가 올 가능성이 높아요”라고 말해줄 수 있어요.

즉, LangChain은 로봇에게  
**“필요하면 책도 찾아보고, 계산기도 쓰고, 문서도 읽어봐!”**  
라고 도와주는 프로그램이에요.

### 이름 뜻

- **Lang**: Language, 즉 “언어”
- **Chain**: “사슬”, 여러 일을 줄줄이 연결한다는 뜻

그래서 LangChain은  
**AI가 여러 일을 순서대로 연결해서 하도록 만드는 도구**라고 볼 수 있어요.

### 예시

학교 도우미 AI를 만든다고 해볼게요.

학생이 이렇게 물어요.

> “우리 학교 급식 메뉴와 내일 날씨를 알려줘.”

LangChain을 이용하면 AI는:

1. 학교 급식표 파일을 읽고
2. 날씨 정보를 인터넷에서 찾고
3. 둘을 합쳐서
4. 친절한 답을 만들어 줍니다.

예를 들면:

> “내일 점심은 카레라이스예요. 오후에는 비가 올 수 있으니 우산을 챙기세요!”

### 한 줄 요약

**LangChain은 AI가 말만 하는 것을 넘어서, 문서·인터넷·계산기 같은 여러 도구를 사용하며 일을 잘하게 도와주는 연결 도구입니다.**
'''