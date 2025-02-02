import os

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

# 调用大语言模型
# 创建模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)
message_weather = HumanMessage(content="北京的天气怎么样?")
message_capital = HumanMessage(content="中国的首都是哪个城市?")

# 没有任何代理的情况下
# res = chatLLM.invoke([message_weather])
# print(res)

# Langchain 内置了一个工具, 可以轻松的使用Tavily搜索引擎作为工具
# max_results: 只返回两个结果
search = TavilySearchResults(max_results=2)
# print(search.invoke("北京的天气怎么样?"))

# 让模型绑定工具
tools = [search]
llm_bind_tools = chatLLM.bind_tools(tools)

# 模型可以自动推理, 是否需要调用工具去完成用户的答案
# res1 = llm_bind_tools.invoke([message_capital])
# print(res1)
# print(f"Model_Result_Content:{res1.content}")
# print(f"Tools_result_Content:{res1.tool_calls}")
#
# res2 = llm_bind_tools.invoke([message_weather])
# print(res2)
# print(f"Model_Result_Content:{res2.content}")
# print(f"Tools_result_Content:{res2.tool_calls}")

# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(chatLLM, tools)
res1 = agent_executor.invoke({"messages": [message_capital]})
print(res1["messages"])
#
res2 = agent_executor.invoke({"messages": [message_weather]})
print(res2["messages"])
print(res2['messages'][2].content)
