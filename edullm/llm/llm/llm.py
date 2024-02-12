

class largeLanguageModel:
    '''define the general llm class properties'''
    def __init__(self, model:str = None, role:str = None, question:str = None) -> None:
        """_summary_

        Args:
            model (str): _description_
            role (str): _description_
            question (str): _description_
        """        
        self.model = model
        self.role = role
        self.question = question

        self.check_model()

    def _get_model(self) -> str:
        return self.model

    def _get_role(self) -> str:
        return self.role

    def _get_question(self) -> str:
        return self.question

    def ask_question(self, question:str) -> str:
        self.question = question

    def set_user(self, user:str) -> str:
        self.user = user

    def check_model(self):
        '''check which model the user wants to use
        #TODO If the model doesn't exist, return an error. If the model can be downloaded, then download it.
        '''
        try:
            assert (self.model == 'mistral' or self.model == "llamaCPP")
        except:
            raise AssertionError('=== Only mistral from ollama and llamaCPP from llamaindex are supported for now ===') 


if __name__ == '__main__':

    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM

    llm = largeLanguageModel(model="llamaCPP")
    llm.ask_question("Why is the sky blue?")
    
    llamaCPP = LlamaIndexLLM().llamaCPP()
    response_iter = llamaCPP.stream_complete(llm._get_question())
    for response in response_iter:
        print(response.delta, end="", flush=True)