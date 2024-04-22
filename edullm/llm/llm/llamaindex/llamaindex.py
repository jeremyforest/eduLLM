from llama_index.llms.llama_cpp import LlamaCPP
from edullm.llm.edu_llm import edu_LLM


class LlamaIndexLLM(edu_LLM):
    def __init__(self) -> None:
        super().__init__(model_type="llamaIndexModel")

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

    def model(self, param_number: int = 7):
        self.model_name = "llamaCPP"

        model_url = f"https://huggingface.co/TheBloke/Llama-2-{param_number}B-chat-GGUF/resolve/main/llama-2-{param_number}b-chat.Q4_0.gguf"

        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 1},
            # transform inputs into Llama2 format
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            verbose=True,
        )
        return llm


if __name__ == "__main__":
    llm = LlamaIndexLLM().model()
    response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
    for response in response_iter:
        print(response.delta, end="", flush=True)
