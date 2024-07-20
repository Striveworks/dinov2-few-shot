import torch
import faiss

from torch import nn

from typing import List


class FaissIndexImageEncoder(nn.Module):
    """
    Uses faiss and a torch model to embed and store
    its inferences.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        dimension: int = 0,
        index_file_init: str = None,
        index_file_out: str = None,
    ):
        """
        Parameters
        ----------
        encoder : nn.Module
          Outputs embeddings in the shape (N, E) where
          N is the batch size and E is the embedding dimension.
        dimension : int
          The dimension of the embeddings
        index_file_init : str
          The index file to start from if it exists.
        index_file_out : str
          The index file to write to.
        """
        super().__init__()

        # TODO: clean up edge logic when an index file is specified
        # to ensure dimensions are the same.

        self.dimension = dimension
        if index_file_init is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.read_index(index_file_init)
        self.index_file_out = index_file_out
        self.metadata = {}

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
        Write a batch of embeddings to the index.

        Parameters
        ----------
        embeddings : torch.Tensor
          A tensor of shape (N, E) where N is the batch size and
          E is the embedding dimension.
        batch_files : List[str]
          The filenames for the images corresponding to each embedding
        """
        self.index.add(embeddings.numpy())

    def flush_to_file(self):
        """
        Flush the index to a flat file.
        """
        faiss.write_index(self.index, self.index_file_out)
