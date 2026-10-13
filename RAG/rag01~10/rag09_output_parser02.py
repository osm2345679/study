# 09-1 카피
# LCEL = Langchain Expression Language
# chian = prompt | model | output_parser

# parser : 분석하다, 답변을 다듬다

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화를 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
- 영어회화 :
- 한글번역 :
"""
# 파이썬 기초 - """ """ 사용하면 문장 길게 쓸 때 한 string으로 취급


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# 06에서 카피함
api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template(template=template)

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url
)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {"question" : "저는 부산에서 물밀면을 먹고 싶어요."}

response = chain.invoke(input)
print(response) # response.content 출력하면 오류남. output_parser가 알아서 결과 가져왔기 때문임
'''
- 영어회화 :  
A: Excuse me, I’d like to try milmyeon in Busan. Do you know a good restaurant?  
B: Sure! There’s a famous milmyeon restaurant nearby.  
A: Great! I’d like to have mulmilmyeon, please.  
B: Good choice. It’s a cold noodle dish with a refreshing broth.  
A: That sounds delicious. Is it spicy?  
B: Not very spicy, but you can add mustard or vinegar if you like.  
A: Perfect. I’ll try it!

- 한글번역 :  
A: 실례합니다, 저는 부산에서 밀면을 먹어 보고 싶어요. 좋은 식당을 아시나요?  
B: 물론이죠! 근처에 유명한 밀면집이 있어요.  
A: 좋아요! 물밀면으로 먹고 싶어요.  
B: 좋은 선택이에요. 시원한 육수가 들어간 차가운 국수 요리예요.  
A: 맛있겠네요. 매운가요?  
B: 많이 맵지는 않지만, 원하시면 겨자나 식초를 넣을 수 있어요.  
A: 완벽해요. 먹어 볼게요!
'''

# gpt-5-nano 결과
'''
- 영어회화 :
A: I’m in Busan and I’d like to try mul naengmyeon.
B: Great choice! Do you want the spicy or mild broth?
A: I’ll have the mild broth, please. Could you tell me what comes with it?
B: It usually comes with noodles, broth, kimchi, and cucumber. Would you like an extra side of pickles or a boiled egg?
A: That sounds good. And could you recommend a way to eat it properly?
B: Definitely. Try loosening the noodles with chopsticks, sip the broth first, and save some broth to wash it all down at the end. 
A: Perfect, I’ll have that. 감사합니다!

- 한글번역 :
A: 부산에 있는데 물밀면 먹고 싶어요.
B: 아주 좋은 선택이에요! 매운 국물로 드릴까요, 아니면 순한 맛으로요?
A: 순한 국물로 주세요. 함께 나오는 구성은 무엇인가요?
B: 보통 면과 국물, 김치, 오이로 구성돼요. 추가로 피클이나 반숙 달걀을 원하나요?
A: 그게 좋겠어요. 그리고 제대로 먹는 방법도 알려주실 수 있나요?
B: 물론이죠. 젓가락으로 면을 풀고 먼저 육수를 한 모금 드신 다음, 나머지를 마무리로 마시면 돼요.
A: 좋습니다. 그렇게 하겠습니다. 감사합니다!
'''