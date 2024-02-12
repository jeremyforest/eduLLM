from llamacpp import Llama


def main():
    llm = Llama(
        model_path="./models/7B/llama-model.gguf",)


if __name__ == '__main__':
    main()