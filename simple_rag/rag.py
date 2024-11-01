import pandas as pd

from langchain.embeddings.base import Embeddings
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker

from .embedding import Embedding


class RAG:

    def __init__(self, data, model) -> None:
        df = pd.DataFrame(data)
        df.head()
        loader = DataFrameLoader(df, page_content_column="question")
        self.documents = loader.load()
        self.model = model

    def query(self, query) -> any:
        embeddings = Embedding(self.model)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )

        docs = text_splitter.split_documents(self.documents)

        chromadb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory="./data"
        )
        chromadb.persist()

        result = chromadb.similarity_search(query)
        return result
