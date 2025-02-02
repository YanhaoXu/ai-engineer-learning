import os
from _operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# 使用LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

# 创建模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 初始化连接mysql数据库的url
HOSTNAME = "127.0.0.1"
PORT = "3306"
DATABASE = "simple_web"
USERNAME = "testdb"
PASSWORD = "123456"

# mysql client的驱动url
MYSQL_URI = ("mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4"
             .format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE))

# 连接数据库
db = SQLDatabase.from_uri(MYSQL_URI)

# 测试链接是否成功
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("select * from t_coffee limit 5;"))

# 直接使用大模型和数据库整合, 只能根据问题生成sql
# 初始化生成SQL的chain
sql_query_chain = create_sql_query_chain(chatLLM, db)
# res = sql_query_chain.invoke({"question": "请问咖啡表中有多少条数据?"})
# print(res)

answer_prompt = PromptTemplate.from_template("""
    给定一下用户问题, SQL语句中的和SQL执行后的结果, 回答用户问题.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答:
    """)
execute_sql_tool = QuerySQLDatabaseTool(db=db)


# 移除返回的SQL的"SQLQuery: "前缀
# 添加一个步骤来移除前缀和````sql```标记
def clean_sql_query(query):
    # 移除"SQLQuery: "前缀
    query = query.replace("SQLQuery: ", "").strip()
    # 移除```sql标记
    if query.startswith("```sql"):
        query = query[7:]  # 去掉开头的```sql
    if query.endswith("```"):
        query = query[:-3]  # 去掉结尾的```
    return query.strip()


remove_prefix_and_markers = RunnableLambda(clean_sql_query)

chain = (
        RunnablePassthrough.assign(query=sql_query_chain)
        .assign(result=itemgetter("query") | remove_prefix_and_markers | execute_sql_tool)
        | answer_prompt
        | chatLLM | StrOutputParser()
)
rep = chain.invoke(input={"question": "请问咖啡表中有多少条数据?"})
print(rep)
