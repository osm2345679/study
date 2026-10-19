from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyData(TypedDict):
    messages : Annotated[list, add_messages]    # add_messages - Reducer

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='hi', id='1')]
msg2 = [AIMessage(content='hello', id='2')]

results = add_messages(msg1, msg2)
print(results)
# [HumanMessage(content='hi', additional_kwargs={}, response_metadata={}, id='1'),AIMessage(content='hello', additional_kwargs={}, response_metadata={}, id='2')]

############################################
msg3 = [HumanMessage(content='hi', id='3')]
msg4 = [AIMessage(content='hello', id='4')]

results = add_messages(msg3, msg4)
print(results)
# [HumanMessage(content='hi', additional_kwargs={}, response_metadata={}, id='3'), AIMessage(content='hello', additional_kwargs={}, response_metadata={}, id='4', tool_calls=[], invalid_tool_calls=[])]