from llama_index.core import SimpleDirectoryReader


class Importer:
    def __init__(self, path: str = "edullm/llm/rag/data/"):
        self.path = path
        self.reader = self.llamaIndexSimpleDirectoryReader()

    def llamaIndexSimpleDirectoryReader(self):
        return SimpleDirectoryReader(input_dir=self.path, recursive=True)

    def load(self, workers: int = 4):
        data = self.reader.load_data(num_workers=workers)
        return data


if __name__ == "__main__":
    importer = Importer()
    documents = importer.load()
    print(len(documents))
    # print(documents)
