__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
#关闭chroma向服务器传输数据，防止程序运行时出现Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given类问题
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import streamlit as st
import load_vectordb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
import sys

sys.path.append("notebook/C3 搭建知识库")  # 将父目录放入系统
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


#创造向量库
@st.cache_resource(show_spinner="正在初始化知识库，请稍候...")
def init_database():
    return load_vectordb.main()

# %%封装检索器
def get_retriever():
    vectordb = init_database()
    return vectordb.as_retriever()


# %%
def combine_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs['context'])


# %%
from zhipuai_LLM import ZhipuaiLLM



def get_qa_history_chain():
    retriever = get_retriever()
    llm = ZhipuaiLLM(
        model_name='GLM-4-Flash-250414',
        api_key=os.environ.get('ZHIPUAI_API_KEY'),
        temperature=0.1
    )
    condense_question_system_template = (
        """请根据聊天记录来回答用户最近的问题，
        如果没有多余的聊天记录则返回用户的问题。"""
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', condense_question_system_template),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}')
        ]
    )
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get('chat_history', False), (lambda x: x['input']) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever
    )
    system_prompt = (
        """你是一个问答任务助手。请使用检索到的上下文片段回答这个问题。
        如果不知道就直接说不知道。
        请使用简介的语言回答用户
        \n\n
        """
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder(variable_name='char_history'),
            ('human', '{input}')
        ]
    )
    qa_chain = (
            RunnablePassthrough().assign(context=combine_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough().assign(context=retrieve_docs).assign(answer=qa_chain)
    return qa_history_chain


# %%
def gen_response(chain, input, char_history):
    response = chain.stream(
        {
            'input': input,
            'char_history': char_history
        }
    )
    for res in response:
        if 'answer' in res.keys():
            yield res['answer']


# %%
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # 保险库，储存对话历史
    if "message" not in st.session_state:
        st.session_state.message = []
    # 储存问答链
    if 'qa_history_chain' not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 搭建聊天窗口
    # 建立容器 高度为500px
    messages = st.container(height=550)
    # 把历史记录加载到聊天框中
    for message in st.session_state.message:
        with messages.chat_message(message[0]):
            st.write(message[1])
    # 接受用户当前的输入
    if prompt := st.chat_input('Say something'):
        st.session_state.message.append(('human', prompt))
        # 显示用户当前的输入
        with messages.chat_message('human'):
            st.write(prompt)
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            char_history=st.session_state.message
        )
        with messages.chat_message('ai'):
            output = st.write_stream(answer)
        st.session_state.message.append(('ai', output))


if __name__ == '__main__':
    main()