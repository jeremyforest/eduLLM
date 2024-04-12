from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from typing import Any, Optional
import os

class Index:
    def __init__(self, name:str):
        self.vector_db_name = name
        self.vector_db_path = f'edullm/llm/rag/db/{name}'
        self.exist = self.check_existing_index()
        # self.storage_context = StorageContext.from_defaults(persist_dir=self.vector_db_path)
    
    def check_existing_index(self) -> bool:
        if os.path.exists(self.vector_db_path):
            exist = True
        else:
            exist = False
        return exist

    def generate_index(self, documents: Any):
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=self.vector_db_path)

    def get_index(self, documents :Optional[Any]):
        if self.exist:
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir = self.vector_db_path))
        else:
            self.generate_index(documents)

if __name__ == "__main__":
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM
    from edullm.llm.rag.embedding import EmbeddingModel
    from llama_index.core import Settings, SimpleDirectoryReader
    
    llm = LlamaIndexLLM().model()
    embedding = EmbeddingModel()
    embedding_model = embedding.model()

    Settings.llm = llm
    Settings.embed_model = embedding_model

    documents = SimpleDirectoryReader(input_files=['edullm/llm/rag/data/paul_graham_essay.txt']).load_data()
    
    index = Index('test')
    index.get_index(documents)
    print(index)
