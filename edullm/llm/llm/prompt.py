from llama_index.core import ChatPromptTemplate



class Prompt:
    def __init__(self):
        self.user= ""
        self.system = ""

    def user_prompt(self, prompt: str):
        self.user = prompt

    def get_user_prompt(self):
        print(self.user)

    def system_prompt(self, prompt: str):
        self.system = prompt

    def get_system_prompt(self):
        print(self.system)


class LlamaIndexPromptFormat(Prompt):
    def __init__(self):
        super().__init__()



if __name__ == "__main__":
    prompt = Prompt()
    user_prompt = prompt.user_prompt('test')
    system_prompt = prompt.system_prompt('test')

    print(prompt.get_user_prompt())
    print(prompt.get_system_prompt())
