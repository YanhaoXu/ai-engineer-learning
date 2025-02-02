import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 使用 LCEL 构建一个简单的 LLM 应用
# 调用大语言模型
# 1. 创建模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 定义提示模板
# 2. 准备prompt
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', "{text}")
])

# 3. 创建返回的数据解析器
parser = StrOutputParser()

# 4. 得到链
chain = prompt_template | chatLLM | parser

# 5. 直接使用chain调用
# print(chain.invoke({"language": "English", "text": "快速的棕色狐狸跳过了懒惰的狗。"}))

app = FastAPI(title="My Langchain Server", version="0.0.1",
              description="使用Langchain翻译任何语句的服务器")

add_routes(app, chain, path="/chain-demo")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8088)
