import os

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 构建RAG对话应用
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

# 创建模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
                                 model="text-embedding-v2", )

# 1. 加载数据: 一篇博客内容数据
loader = WebBaseLoader(web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('post-header', 'post-title', 'post-content'))))

docs = loader.load()

# print(len(docs))
# print(docs)

# 2. 大文本的切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
split_documents = splitter.split_documents(docs)

# 3. 存储
vector_store = Chroma.from_documents(documents=split_documents, embedding=embeddings)

# 3. 检索器
retriever = vector_store.as_retriever()

# 整合

# 创建一个问题的模板
# prompt = hub.pull("rlm/rag-prompt")


system_prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""

"""
system_prompt_template 的翻译:
您是问答任务的助手。
使用以下检索到的上下文来回答
问题。如果你不知道答案，就说你
不知道。最多使用三个句子并保持答案简洁。

{context}
"""

# 提问和回答的历史记录模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        MessagesPlaceholder("chat_history"),  # ???
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatLLM, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# response = rag_chain.invoke({"input": "What is Task Decomposition?"})
# print(response["answer"])

"""
一般情况下, 我们构建的链(chain)直接使用输入问答记录来关联上下文. 但在此案例中, 查询检索器也需要对话上下文才能被理解.

解决方法:
添加一个子链(chain), 它采用最新用户问题和聊天历史, 并在它引用历史信息中的任何信息时重新表述问题.
这可以被简单的认为是构建一个新的"历史感知"检索器.
这个子链的目的: 让检索过程融入了对话的上下文
"""

# 子链的提示模板
condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
"""
condense_question_system_template 的翻译:
给定一段聊天记录和最新的用户提问，该提问可能涉及聊天记录中的上下文，请构造一个独立的问题，
使其无需查阅聊天记录也能被理解。请勿回答问题，仅需在必要时重新表述问题，否则直接返回原问题。
"""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(chatLLM, retriever, retriever_history_temp)

# 保持问答的历史记录
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建父chain: 把前两个链整合
chain = create_retrieval_chain(history_chain, question_answer_chain)
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# # 第一轮对话
rep1 = result_chain.invoke({"input": "What is Task Decomposition?"},
                           config={"configurable": {"session_id": "saber101"}})
print("第一轮对话:任务分解是什么?")
print(rep1["answer"])

# 第二轮对话
rep2 = result_chain.invoke({"input": "What are common ways of doing it?"},
                           config={"configurable": {"session_id": "saber101"}})
print("第二轮对话:常见的任务分解方法有哪些?")
print(rep2["answer"])
