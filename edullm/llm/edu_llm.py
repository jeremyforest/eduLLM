class edu_LLM:
    """define the general llm class properties"""

    def __init__(
        self,
        model_type: str = "",
        model_name: str = "",
        role: str = "",
        question: str = "",
    ) -> None:
        """ """

        self.model_type = model_type
        self.model_name = model_name
        self.role = role
        self.question = question

    def _checks(self):
        """Checking that the requirement to run the model,
        whatever they are, are present
        """
        self.check_model()

    def _get_model_type(self) -> str:
        return self.model_type

    def _get_model_name(self) -> str:
        return self.model_name

    def _get_role(self) -> str:
        return self.role

    def _get_question(self) -> str:
        return self.question

    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

        return prompt

    def completion_to_prompt(self, completion):
        return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

    def define_question(self, question: str):
        self._checks()
        self.question = question

    def set_user(self, user: str):
        self.user = user

    def check_model(self):
        """check which model the user wants to use
        #TODO If the model doesn't exist, return an error.
        If the model can be downloaded, then download it.
        """
        try:
            assert self.model == "mistral" or self.model == "llamaCPP"
        except AssertionError:
            raise AssertionError(
                """=== Only mistral from ollama and llamaCPP from '
                llamaindex are supported for now ==="""
            )

    def model(self):
        """which model to use. This is defined by children classes"""
        pass


if __name__ == "__main__":
    from edullm.llm.llm.llamaindex.llamaindex import LlamaIndexLLM

    llm = LlamaIndexLLM()
    print(llm.model_type)
    llm = llm.define_question("Why is the sky blue?")
    model = llm.model()
    response_iter = model.stream_complete(llm._get_question())
    for response in response_iter:
        print(response.delta, end="", flush=True)
