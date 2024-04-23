from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EmbeddingModel:
    def __init__(
        self,
        rag_folder: str = "edullm/llm/rag",
        emb_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.rag_folder = rag_folder
        self.rag_data_folder = (f"{self.rag_folder}/data",)
        self.emb_model = emb_model

        # self.embedding_model()

    def model(self):
        """Create the embedding model"""
        print(f"=== using {self.emb_model} for embedding ===")
        model = HuggingFaceEmbedding(model_name=f"{self.emb_model}")
        return model


def main():
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM
    from edullm.llm.llm.ollama.o_llama import OllamaLLM

    # llm = LlamaIndexLLM().model()
    llm = OllamaLLM()
    llm.pull_model()
    llm.launch_server()
    llm_model = llm.model()
    embedding = EmbeddingModel()
    embedding_model = embedding.model()

    from llama_index.core import Settings

    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    # load documents
    # documents = SimpleDirectoryReader(embedding.rag_data_folder[0]).load_data()

    from edullm.inputs.importer import Importer

    documents = Importer().load()

    # create vector store index
    # index = VectorStoreIndex.from_documents(documents, show_progress=True)
    from edullm.llm.rag.vector_database import Index
    index = Index("neuromatch").get_index(documents)


    # set up query engine
    streaming = False
    query_engine = index.as_query_engine(streaming=streaming)

    response = query_engine.query(
        "Can you summarize what the data I provided with is about in a concise paragram of 300 words ?"
    )
    if streaming:
        [print(token) for token in response.response_gen]
    else:
        print(response)


if __name__ == "__main__":
    main()
