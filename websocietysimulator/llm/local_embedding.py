import requests
from typing import List


class LocalEmbedding:
    """
    Local embedding model using Ollama's embedding API.

    This class replaces OpenAIEmbeddings and is fully compatible 
    with MemoryGenerative's usage pattern:
       - embed_query(text)
       - embed_documents([text1, text2, ...])
    """

    def __init__(self, model: str = "nomic-embed-text"):
        """
        Args:
            model: The embedding model to use (must be available in Ollama).
                   Recommended: "nomic-embed-text"
        """
        self.model = model
        self.url = "http://localhost:11434/api/embed"


    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text input into a vector.

        Args:
            text: A single string.

        Returns:
            A list of floats representing the embedding vector.
        """
        payload = {
            "model": self.model,
            "input": text
        }

        res = requests.post(self.url, json=payload)
        res.raise_for_status()

        # Ollama returns {"embedding": [...]} for a single input
        return res.json()["embedding"]


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts into vectors.

        Args:
            texts: List of strings.

        Returns:
            A list of embedding vectors (one per text).
        """
        payload = {
            "model": self.model,
            "input": texts
        }

        res = requests.post(self.url, json=payload)
        res.raise_for_status()

        # Ollama returns {"embeddings": [[...], [...], ...]}
        return res.json()["embeddings"]
