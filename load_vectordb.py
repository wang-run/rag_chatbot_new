#导入api
import os
from dotenv import load_dotenv


def load_api():
    return load_dotenv('demo.env')


#%%
#导入txt文件将其变为document对象

from langchain_community.document_loaders import TextLoader
def txt_to_document(txt_path):
    """
    :param txt_path:
    :return:docs, txt转化成document对象
    """
    loader = TextLoader(txt_path, autodetect_encoding=True)
    return loader.load()

#%%
#对文本进行切分
from langchain_text_splitters import CharacterTextSplitter
def split_document(documents):
    """
    :param documents:
    :return:split_docs, 切分之后放在列表中
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=50,
        chunk_overlap=0
    )
    return text_splitter.split_documents(documents)
#%%
#将切割好的文本进行向量化并封装到向量库中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
def save_vectordb(split_docs, vectordb_path = 'vectordb/chroma'):
    embedding = ZhipuAIEmbeddings()  # 创建embedding对象
    persist_directory = vectordb_path  # 指定向量库存放位置
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb




#查看文件是否存在，防止文件存在的情况下chroma.from_message对文本多次最加
def check_file_exists(file_path):
    #检测路径是否存在
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            return True
        else:
            return False
    else:
        return False

def main():
    load_api()
    vectordb_path = 'vectordb/chroma'
    txt_path = 'data_base/organize_sentence.txt'
    if not check_file_exists(vectordb_path):
        docs = txt_to_document(txt_path)
        split_docs = split_document(docs)
        vectordb = save_vectordb(split_docs)
        return vectordb
    else:
        embedding = ZhipuAIEmbeddings()
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory= vectordb_path
        )
        return vectordb

def text():
    question = '梁磊是什么专业的？'
    vectordb = main()
    retriever = vectordb.as_retriever(search_kwargs={'k': 2})
    sim_answer = retriever.invoke(question)
    for i, answer in enumerate(sim_answer):
        print(f"第{i}个相似的回答为：{answer.page_content}")

#
if __name__ == '__main__':
    main()
    text()

