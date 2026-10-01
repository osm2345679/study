# LCEL = Langchain Expression Language
# chian = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()
# vscode에서는 한 번 입력하면 알아서 계속 주입하거나 알아서 .env 파일 열어서 주입해주지만 ide 아니면 적용 안 되므로 꼭 적어줄 것.

# 06에서 카피함
api_key = os.environ["MONOROUTER_API_KEY"].strip()   # strip은 공백이나 줄바꿈 등 제거해주는 메서드
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요.")

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url
)

chain = prompt | model

input = {"topic" : "양자컴퓨터 학습 원리"}

# response = chain.invoke(input)
# print(response)
# content='양자컴퓨터의 원리는 쉽게 말하면 **“0과 1만 사용하는 일반 컴퓨터보다, 여러 가능성을 동시에 다루고 그중 정답 가능성을 크게 만드는 계산 방식”**입니다.\n\n## 1. 일반 컴퓨터: 비트(bit)\n\n일반 컴퓨터는 정보를 **비트**로 저장합니다.\n\n- `0` 또는 `1`\n- 전등으로 비유하면: 꺼짐(0), 켜짐(1)\n\n예를 들어 3비트가 있으면 한 순간에 아래 중 **하나의 상태**만 가집니다.\n\n- 000\n- 001\n- 010\n- …\n- 111\n\n---\n\n## 2. 양자컴퓨터: 큐비트(qubit)\n\n양자컴퓨터는 **큐비트**를 사용합니다.\n\n큐비트는 측정하기 전까지는 단순히 0 또는 1 중 하나가 아니라,\n\n> **0일 가능성과 1일 가능성을 함께 가진 상태**\n\n가 될 수 있습니다. 이를 **중첩(superposition)**이라고 합니다.\n\n동전을 예로 들면:\n\n- 바닥에 놓인 동전: 앞면 또는 뒷면 → 일반 비트\n- 공중에서 빙글빙글 도는 동전: 앞면과 뒷면 가능성이 함께 존재 → 큐비트의 중첩 상태\n\n단, 동전을 잡아 확인하는 순간 앞면 또는 뒷면 하나로 결정되듯, 큐비트도 **측정하면 0이나 1 중 하나로 결과가 정해집니다.**\n\n---\n\n## 3. 여러 가능성을 동시에 표현하는 이유\n\n큐비트가 늘어나면 표현할 수 있는 상태 수가 매우 빠르게 증가합니다.\n\n| 큐비트 수 | 동시에 표현 가능한 상태 수 |\n|---:|---:|\n| 1개 | 2개 |\n| 2개 | 4개 |\n| 3개 | 8개 |\n| 10개 | 1,024개 |\n| 50개 | 약 1,000조 개 |\n\n그래서 양자컴퓨터는 복잡한 경우의 수를 다루는 데 잠재력이 있습니다.\n\n다만 중요한 점은:\n\n> 양자컴퓨터가 모든 답을 한꺼번에 읽어낼 수 있는 것은 아닙니다.\n\n측정하면 결국 한 결과만 얻습니다. 따라서 양자 알고리즘의 핵심은 단순히 “여러 답을 동시에 계산”하는 것이 아니라, **정답의 확률은 높이고 오답의 확률은 낮추도록 계산을 설계하는 것**입니다.\n\n---\n\n## 4. 얽힘(entanglement): 큐비트끼리 연결되는 현상\n\n양자컴퓨터의 또 다른 핵심은 **얽힘**입니다.\n\n두 큐비트가 얽히면, 하나를 측정했을 때 다른 하나의 상태도 강하게 연결됩니다.\n\n예를 들어 두 큐비트가 다음처럼 얽힐 수 있습니다.\n\n- 첫 번째가 0이면 두 번째도 0\n- 첫 번째가 1이면 두 번째도 1\n\n측정 전에는 둘 다 확정되지 않았지만, 하나를 확인하면 다른 하나와의 관계가 즉시 드러납니다.\n\n이 얽힘 덕분에 양자컴퓨터는 여러 변수 사이의 복잡한 관계를 표현하고 처리할 수 있습니다.\n\n---\n\n## 5. 간섭(interference): 정답을 강화하는 핵심\n\n양자컴퓨터가 실제로 문제를 푸는 핵심 원리는 **간섭**입니다.\n\n파도가 만나면 다음 두 가지가 일어납니다.\n\n- 파도끼리 겹쳐 더 커짐 → **보강 간섭**\n- 서로 상쇄되어 작아짐 → **상쇄 간섭**\n\n양자 알고리즘도 비슷합니다.\n\n- 정답으로 이어지는 계산 경로는 서로 강화\n- 오답으로 이어지는 계산 경로는 서로 상쇄\n\n되도록 만듭니다.\n\n즉, 양자컴퓨터는 마지막 측정 시점에 **정답이 나올 확률을 높이는 방식**으로 작동합니다.\n\n---\n\n## 6. 양자 게이트: 큐비트를 조작하는 명령어\n\n일반 컴퓨터에는 AND, OR, NOT 같은 논리 회로가 있습니다.  \n양자컴퓨터에는 **양자 게이트(quantum gate)**가 있습니다.\n\n대표적인 예:\n\n- **X 게이트**: 0과 1을 바꿈  \n  - 0 → 1\n  - 1 → 0\n\n- **H 게이트(Hadamard gate)**: 0 또는 1을 중첩 상태로 만듦\n\n- **CNOT 게이트**: 두 큐비트를 연결해 얽힘을 만드는 데 사용\n\n양자 알고리즘은 이런 게이트를 정해진 순서로 배치한 **양자 회로**입니다.\n\n---\n\n## 7. “학습”은 어떻게 하나요?\n\n만약 질문의 “학습”이 인공지능처럼 데이터를 학습하는 것을 뜻한다면, 양자컴퓨터도 비슷한 방식으로 쓸 수 있습니다.\n\n### 일반 AI 학습\n1. 데이터 입력\n2. 모델이 예측\n3. 정답과 비교\n4. 오차를 줄이도록 모델의 파라미터 조정\n5. 반복\n\n### 양자 머신러닝\n1. 데이터를 큐비트 상태로 표현\n2. 양자 게이트로 데이터를 변환\n3. 측정 결과를 얻음\n4. 결과가 좋아지도록 게이트의 설정값(파라미터)을 조정\n5. 반복\n\n이때 양자컴퓨터는 복잡한 데이터 패턴, 분자 구조, 최적화 문제 등에서 일부 장점을 낼 가능성이 연구되고 있습니다.\n\n하지만 현재는 아직 초기 단계이며, 모든 AI 문제를 양자컴퓨터가 더 잘 푸는 것은 아닙니다.\n\n---\n\n## 8. 한 줄 요약\n\n양자컴퓨터는\n\n> **중첩으로 여러 가능성을 표현하고, 얽힘으로 큐비트들을 연결하며, 간섭으로 정답 가능성을 키워 문제를 푸는 컴퓨터**\n\n입니다.\n\n그리고 “학습”에서는 이 양자적 성질을 이용해 데이터의 복잡한 패턴을 찾아내거나, 최적의 해를 더 효율적으로 찾는 방법을 연구합니다.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 1471, 'prompt_tokens': 19, 'total_tokens': 1490, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 80, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETgQB4YNkdcUZa0qGPKKlK0XApgif', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0f09e-15c0-7293-b34d-ca50a68d730c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 19, 'output_tokens': 1471, 'total_tokens': 1490, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 80}}

input = {"topic" : "lcel"}
response = chain.invoke(input)

print(response.content)

'''
LCEL은 **LangChain Expression Language**의 약자로, LangChain에서 AI 작업 흐름을 **간단한 문법으로 연결해서 만드는 방법**입니다.


쉽게 말하면:

> “프롬프트 만들기 → AI에게 질문하기 → 답변 형식 정리하기”  
> 같은 단계를 파이프(`|`)로 이어 붙이는 방식입니다.

## 왜 필요한가요?

AI 앱은 보통 한 번에 끝나지 않습니다.

예를 들어 사용자의 질문을 받으면:

1. 질문을 프롬프트 형식으로 바꾸고
2. LLM(GPT 등)에 전달하고
3. 나온 결과를 문자열이나 JSON으로 정리해야 합니다.

LCEL은 이 과정을 보기 좋게 연결해 줍니다.

---

## 기본 모습

```python
chain = prompt | model | parser
```

의미는 다음과 같습니다.

- `prompt`: 질문을 AI가 이해할 프롬프트로 만듦
- `model`: GPT 같은 언어 모델 호출
- `parser`: AI의 응답을 원하는 형태로 변환

즉, 왼쪽 결과가 오른쪽 입력으로 전달됩니다.

마치 리눅스 파이프나 데이터 처리 파이프라인과 비슷합니다.

---

## 간단한 예시

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "{topic}을 초등학생도 이해할 수 있게 설명해줘."
)

model = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "블랙홀"})

print(result)
```

이 코드에서는:

```python
{"topic": "블랙홀"}
```

이라는 입력이 들어가고,

1. 프롬프트가 만들어집니다.
   - “블랙홀을 초등학생도 이해할 수 있게 설명해줘.”
2. 모델이 답변합니다.
3. `StrOutputParser()`가 답변을 일반 문자열로 꺼내 줍니다.

---

## `invoke()`는 무엇인가요?

LCEL 체인을 실행할 때 자주 씁니다.

```python
chain.invoke({"topic": "인공지능"})
```

- 한 번 요청하고
- 한 번 결과를 받습니다.

또한 여러 입력을 한꺼번에 처리하거나 실시간 출력도 가능합니다.

```python
chain.batch([
    {"topic": "인공지능"},
    {"topic": "양자컴퓨터"},
])
```

```python
for chunk in chain.stream({"topic": "우주"}):
    print(chunk, end="")
```

- `batch()`: 여러 요청을 묶어서 실행
- `stream()`: 답변이 생성되는 대로 조금씩 받기

---

## 여러 작업을 함께 처리할 수도 있습니다

예를 들어 질문에 대해 “답변”과 “난이도 평가”를 동시에 만들 수 있습니다.

```python
chain = {
    "answer": prompt | model | StrOutputParser(),
    "length": prompt | model | StrOutputParser(),
}
```

LCEL은 이런 식으로 작업을 **순차 연결**, **병렬 처리**, **분기 처리**하기 편하게 해 줍니다.

---

## 핵심만 정리하면

LCEL은 LangChain에서 AI 처리 단계를 조립하는 문법입니다.

```python
입력 | 프롬프트 | AI 모델 | 결과 정리
```

장점은 다음과 같습니다.

- 코드가 짧고 읽기 쉽습니다.
- 프롬프트, 모델, 출력 처리 단계를 재사용하기 쉽습니다.
- 스트리밍, 배치 처리, 비동기 실행 등을 지원합니다.
- 복잡한 AI 워크플로우를 파이프라인 형태로 관리할 수 있습니다.

한 줄로 표현하면:

> **LCEL은 LangChain에서 AI 기능들을 `|`로 연결해 작업 흐름을 만드는 방식입니다.**
'''