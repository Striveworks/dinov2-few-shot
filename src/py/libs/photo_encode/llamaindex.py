import torch

from torch import nn
from llama_index import LlamaIndex, Node

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
        self.index = LlamaIndex(nodes=[])
        self.index_file = llama_index_config["index_file"]

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
        nodes = [
            Node(id=file_path, data=embeddings[i].numpy())
            for i, file_path in enumerate(batch_files)
        ]
        self.index.nodes.extend(nodes)
        self.index.save(self.index_file)
