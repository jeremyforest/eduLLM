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

        # self.embedding_model()
        
    def model(self):
        """ Create the embedding model
        """
        print(f"=== using {self.emb_model} for embedding ===")
        model = HuggingFaceEmbedding(model_name=f"{self.emb_model}")
        return model
    

def main():
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM
    llm = LlamaIndexLLM().model()
    embedding = EmbeddingModel()
    embedding_model = embedding.model()

    from llama_index.core import Settings
    Settings.llm = llm
    Settings.embed_model = embedding_model

    # load documents
    documents = SimpleDirectoryReader(embedding.rag_data_folder[0]).load_data()
    
    # create vector store index
    index = VectorStoreIndex.from_documents(documents)

    # set up query engine
    streaming = False
    query_engine = index.as_query_engine(streaming=streaming)

    response = query_engine.query("What did the author do growing up?")
    if streaming:
        [print(token) for token in response.response_gen]
    else:
        print(response)


if __name__ == '__main__':
    main()
