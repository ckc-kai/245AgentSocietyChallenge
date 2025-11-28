import requests
from typing import List, Dict, Optional, Union
from .llm import LLMBase
from .local_embedding import LocalEmbedding


class LocalLLM(LLMBase):
    """
    Local LLM using Ollama + local embedding model.
    Fully compatible with Simulator, MemoryGenerative, ReasoningCOT.
    """

    def __init__(self, model: str = "llama3.1:8b", embedding_model_name="nomic-embed-text"):
        super().__init__(model)
        self.url = "http://localhost:11434/api/chat"
        self.embedding_model = LocalEmbedding(embedding_model_name)   


    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # support optional stop sequences (MemoryGenerative sometimes uses these)
        if stop_strs:
            payload["options"]["stop"] = stop_strs

        res = requests.post(self.url, json=payload)
        res.raise_for_status()

        return res.json()["message"]["content"]


    def get_embedding_model(self):
        return self.embedding_model
