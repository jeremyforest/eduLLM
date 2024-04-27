from llama_index.llms.ollama import Ollama
from edullm.llm.edu_llm import edu_LLM
from ollama import Client
import ollama


class OllamaLLM(edu_LLM):
    def __init__(self, model_name: str = 'phi3') -> None:
        super().__init__(model_type="OllamaModel")
        self.model_name = model_name

    def pull_model(self):
        '''If the model has never been used, it needs to be pulled
        '''
        ollama.pull(self.model_name)

    def launch_server(self):
        client = Client(host='http://localhost:11434')

        # empty content just load the model in memory
        response = client.generate(model=self.model_name, prompt='')
        return response

    def model(self):
        """ Load the model specified Ollama model and return it wrapped
        throught llamaIndex
        """
        llm = Ollama(model=self.model_name, request_timeout=300)
        return llm


if __name__ == "__main__":

    llm = OllamaLLM()
    llm.pull_model()
    llm.launch_server()
    llm_model = llm.model()
    response_iter = llm_model.stream_complete(
        "Can you write me a poem about fast cars?")
    for response in response_iter:
        print(response.delta, end="", flush=True)
