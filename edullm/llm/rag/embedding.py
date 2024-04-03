from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class EmbeddingModel:
    def __init__(self, 
                rag_folder:str = "edullm/llm/rag",
                emb_model:str = "BAAI/bge-small-en-v1.5"
                ):
        self.rag_folder = rag_folder
        self.rag_data_folder = f"{self.rag_folder}/data",
        self.emb_model = emb_model

        self.embedding_model()
        
    def embedding_model(self):
        """ Create the embedding model
        """
        model = HuggingFaceEmbedding(model_name=f"{self.emb_model}")
        return model
    

def main():
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM
    llm = LlamaIndexLLM().model()
    embedding = EmbeddingModel()


    from llama_index.core import Settings
    Settings.llm = llm
    Settings.embed_model = embedding

    # load documents
    documents = SimpleDirectoryReader(embedding.rag_data_folder[0]).load_data()
    
    # create vector store index
    index = VectorStoreIndex.from_documents(documents)

    # set up query engine
    query_engine = index.as_query_engine()

    response = query_engine.query("What did the author do growing up?")
    print(response)


if __name__ == '__main__':
    main()
