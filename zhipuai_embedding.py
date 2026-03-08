from langchain_core.embeddings import Embeddings
from typing import List


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI()
    #定义对文本进行嵌入，许多api接口对token有限制，这里进行了拆分，最大长度为60
    #一轮一轮的调用api
    def embed_documents(self, text: List[str]) -> List[List[float]]:
        batch_size = 60
        all_embeddings = []
        for i in range(0, len(text), batch_size):
            batch_text = text[i: i + batch_size]
            response = self.client.embeddings.create(
                model='embedding-3',
                input=batch_text
            )
            batch_embeddings = [embeddings.embedding for embeddings in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str):
        response = self.client.embeddings.create(
            model='embedding-3',
            input=text
        )
        return response.data[0].embedding