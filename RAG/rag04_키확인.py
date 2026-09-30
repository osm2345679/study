import os
key = os.getenv('OPENAI_API_KEY')   # 환경변수 가져오기

if key is None:
    print("OPEN_API_KEY가 없음")
else:
    print("키 길이 :", len(key))
    print("키 확인 :", key[:8] + "..." + key[-4:])
# 키 길이 : 164
# 키 확인 : sk-proj-...KecA

