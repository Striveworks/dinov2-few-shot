import torch
import faiss

from torch import nn
from llama_index.core import VectorStoreIndex

from typing import List


class LLaMAIndexImageEncoder(nn.Module):
    """
    Uses llama-index and a torch model to embed and store
    its inferences.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        llama_index_config: dict = None,
    ):
        """
        Parameters
        ----------
        encoder : nn.Module
          Outputs embeddings in the shape (N, E) where
          N is the batch size and E is the embedding dimension.
        llama_index_config : dict
          The configuration for the indexer
        """
        super().__init__()
        self.index_file = llama_index_config["index_file"]
        self.dimension = llama_index_config["embedding_dimension"]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.vector_store_index = VectorStoreIndex(self.index)

    def forward(self, x: torch.Tensor, batch_files: List[str]):
        """
        Overload the forward method to store the
        embeddings with the indexer.

        Parameters
        ----------
        x : torch.Tensor
          The input tensor
        batch_files : List[str]
          The filenames for the images corresponding to each embedding

        Returns
        -------
        out : torch.Tensor
          The original output from the encoder model.
        """
        # Get the embedding predictions
        out = super().forward(x)
        self.store_embeddings(out, batch_files)
        return out

    def store_embeddings(self, embeddings: torch.Tensor, batch_files: List[str]):
        """
        Write a batch of embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
          A tensor of shape (N, E) where N is the batch size and
          E is the embedding dimension.
        batch_files : List[str]
          The filenames for the images corresponding to each embedding
        """
        self.vector_store_index.add(embeddings.numpy())
