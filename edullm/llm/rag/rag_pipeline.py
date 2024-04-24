
from edullm.llm.llm.ollama.o_llama import OllamaLLM
from edullm.llm.rag.vector_database import Index
from edullm.inputs.importer import Importer
from edullm.llm.rag.embedding import EmbeddingModel
from edullm.llm.llm.prompt import Prompt
from llama_index.core import Settings, PromptTemplate


class RAGPipeline:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.documents = None
        self.index_database = None

        self._llm()
        self._embedding()
        self._import_documents()
        self._load_index_database()

    def _llm(self):
        """ Which llm to use for interaction with the documents
        """
        self.llm = OllamaLLM()
        self.llm.pull_model()
        self.llm.launch_server()
        llm_model = self.llm.model()
        Settings.llm = llm_model

    def _embedding(self):
        """ Define which Embedding Model will be used
        """
        self.embedding = EmbeddingModel()
        embedding_model = self.embedding.model()
        Settings.embed_model = embedding_model

    def _import_documents(self):
        """ Fetch the documents through the indexed db
        """
        self.documents = Importer().load()

    def _load_index_database(self, name: str = 'neuromatch'):
        """ Load the database
        """
        self.index_database = Index(name).get_index(self.documents)

    def _query_chat_engine(self, query, mode: str = 'chat'):
        """ Query in one shot or converse with the LLM model about the data
        """
        # custom_prompt = self._prompt_customization()
        if mode == 'query':
            query_engine = self.index_database.as_query_engine()
            # query_engine.update_prompts(
            #         {"response_synthesizer:summary_template": custom_prompt}
            # )
            response = query_engine.query(query)
        elif mode == 'chat':
            chat_engine = self.index_database.as_chat_engine()
            # query_engine.update_prompts(
            #         {"response_synthesizer:summary_template": custom_prompt}
            # )
            response = chat_engine.chat(query)
        return response

#     def _prompt_customization(self) -> PromptTemplate:
#        prompt = Prompt()
#        user_prompt = prompt.set_system_prompt()
#        return PromptTemplate(user_prompt)


if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    while True:
        query = input('Enter your question: \n')
        response = rag_pipeline._query_chat_engine(query)
        print(response)
