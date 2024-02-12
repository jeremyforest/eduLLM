from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_tokenizer
# from llama_index.embeddings import OptimumEmbedding
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer


set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)


class EmbeddingModel:
    def __init__(self, 
                rag_folder:str = "edullm/llm/rag",
                embedding_model:str = ""
                ):
        self.rag_folder = rag_folder
        self.rag_data_folder = f"{self.rag_folder}/data",
        self.embedding_model = embedding_model

    def embed_model(self, model:str = 'bge-small-en-v1.5'):
        """ Create the embed model
        """
        model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return model
    

def main():
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM
    llm = LlamaIndexLLM().llamaCPP()
    embedding = EmbeddingModel()
    embed_model = embedding.embed_model()

    # create a service context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )

    # load documents
    documents = SimpleDirectoryReader(embedding.rag_data_folder[0]).load_data()
    
    # create vector store index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # set up query engine
    query_engine = index.as_query_engine()

    response = query_engine.query("What did the author do growing up?")
    print(response)


if __name__ == '__main__':
    main()
