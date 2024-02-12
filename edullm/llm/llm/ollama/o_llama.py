import ollama 
from edullm.llm.llm.llm import largeLanguageModel


llm = largeLanguageModel(model="mistral", role="user", question='Why is the sky blue?')

def chat_stream():
    # from https://github.com/ollama/ollama-python/tree/main/examples
    stream = ollama.chat(
        model=llm.model,
        messages=[{'role': llm.role, 'content': llm.question}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def main():
    chat_stream()

if __name__ == '__main__':
    main()
