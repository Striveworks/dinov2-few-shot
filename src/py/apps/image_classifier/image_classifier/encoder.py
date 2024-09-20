import torch
import faiss
import numpy as np

from torch import nn
from copy import deepcopy

from typing import List, Union, Any, Callable


class FaissIndexImageEncoder(nn.Module):
    """
    Uses faiss and a torch model to embed and store
    its inferences.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        index_file_out: str = None,
    ):
        """
        Parameters
        ----------
        encoder : nn.Module
          Outputs embeddings in the shape (N, E) where
          N is the batch size and E is the embedding dimension.
        index_file_out : str
          The index file to write to.
        """
        super().__init__()

        # TODO: clean up edge logic when an index file is specified
        # to ensure dimensions are the same.

        self.index_file_out = index_file_out
        self.metadata = {}
        self.encoder = encoder
        self.current_embeddings = None
        self.current_files = None

    def forward(
        self,
        x: Union[torch.Tensor, dict],
        batch_files: List[str],
        process_embeddings: Callable[[Any], Any],
    ):
        """
        Overload the forward method to store the
        embeddings with the indexer.

        Parameters
        ----------
        x : torch.Tensor
          The input tensor
        batch_files : List[str]
          The filenames for the images corresponding to each embedding
        process_embeddings : Callable[Any, Any]
          Post-process the outputs

        Returns
        -------
        out : torch.Tensor
          The original output from the encoder model.
        """
        # Get the embedding predictions
        if isinstance(x, dict):
            out = self.encoder.forward(**x)
        else:
            out = self.encoder.forward(x)
        out_processed = process_embeddings(out)

        # Keep a running list of our embeddings
        if self.current_embeddings is None:
            self.current_embeddings = out_processed.cpu()
        else:
            self.current_embeddings = torch.cat(
                (self.current_embeddings, out_processed.cpu())
            )

        # Keep a running list of filenames
        if self.current_files is None:
            self.current_files = deepcopy(batch_files)
        else:
            self.current_files += batch_files

        return out

    def flush_to_file(
        self, dimension=1024, n_centroids=2, n_subquantizers=2, code_size=2
    ):
        """
        Flush the index to a flat file.

        TODO: clean up the training logic. Not currently functional.
        """
        self.index.train(self.current_embeddings)
        faiss.write_index(self.index, self.index_file_out)
