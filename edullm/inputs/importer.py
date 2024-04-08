from llama_index.core import SimpleDirectoryReader

class Importer():
    def __init__(self, path:str = 'edullm/llm/rag/data'):
        self.path = path
        self.reader = self.llamaIndexSimpleDirectoryReader()

    def llamaIndexSimpleDirectoryReader(self):
        return SimpleDirectoryReader(input_dir=self.path)

    def load(self, workers:int = 4):
        docs = self.reader.load_data(num_workers=workers)
        return docs

if __name__ == '__main__':
    importer = Importer()
    print(importer.path)
    print(importer.reader)
    documents = importer.load()
    print(len(documents))
    # print(documents)
