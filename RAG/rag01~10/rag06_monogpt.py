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
    api_key=api_key,
    base_url=base_url
)

# response = llm.invoke('안녕')
# print(response)
# content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 8, 'total_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLhyVgBpkh8vwBzvrihFL7WzkDh', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-e2bc-7d90-ae8d-87fd00d6a62d-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 8, 'output_tokens': 14, 'total_tokens': 22, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
# print(response.content) # 안녕하세요! 무엇을 도와드릴까요?

# 메모리 기능 등이 없어서 이전 대화 내용 모르고 매번 새 대화임