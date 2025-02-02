import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

# 聊天机器人案例
# 创建模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 定义提示词模板
chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "你是一个乐于助人的助手, 用{language}尽你所能回答所有的问题."),
     MessagesPlaceholder(variable_name="my_msg")])
# 得到链
chain = chat_prompt_template | chatLLM

# 保存聊天记录
# 所有用户的聊天记录都保存到store
# key: sessionId, value: 历史聊天记录对象
store = {}


# 此函数预期将接收一个session_id并返回一个历史消息记录对象
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="my_msg")

# 给当前会话定义一个session_id
config = {"configurable": {"session_id": "test001"}}

# 第一轮
res01 = do_message.invoke(
    {"my_msg": [HumanMessage(content="你好! 我是Saber")], "language": "中文"}, config=config)

print(res01.content)

# 第二轮
res02 = do_message.invoke(
    {"my_msg": [HumanMessage(content="请问我是谁?")], "language": "中文"}, config=config)

print(res02.content)

# 第三轮, 返回的数据是流式的
config = {"configurable": {"session_id": "test002"}}

for res in do_message.stream(
        {"my_msg": [HumanMessage(content="请问给我讲一个笑话?")], "language": "English"},
        config=config):
    print(res.content, end="-")
