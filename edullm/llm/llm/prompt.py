from llama_index.core import ChatPromptTemplate
from llama_index.core import PromptTemplate


class Prompt:
    def __init__(self):
        self.user_prompt = ""
        self.system_prompt = ""

    def set_user_prompt(self):
        self.user_prompt = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query in a consise way attempting to teach, "
            "a university graduate course level.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

    def get_user_prompt(self):
        print(self.user_prompt)

    def set_system_prompt(self):
        self.system_prompt = () 

    def get_system_prompt(self):
        print(self.system_prompt)


if __name__ == "__main__":
    prompt = Prompt()

    print(prompt.get_user_prompt())
    print(prompt.get_system_prompt())

    user_prompt = prompt.set_user_prompt()
    system_prompt = prompt.set_system_prompt()

    print(prompt.get_user_prompt())
    print(prompt.get_system_prompt())
