"""
Generic utilities used across the library.
"""

import torch
import os

from typing import Callable, List
from PIL import Image


def image_batch_from_folder_generator(
    folder: str, transform: Callable[Image, torch.Tensor], batch_size: int
) -> (torch.Tensor, List[str]):
    """
    Loads images from a folder as tensor batches until the
    folder is empty.

    Parameters
    ----------
    folder : str
      The local folder to pull images from
    transform : Callable
      Produces a torch tensor from PIL images. For example, a Compose transform.
    batch_size : int
      The batch size

    Yields
    ------
    x : torch.Tensor
      A tensor of shape (N, C, W, H) where N is the batch size, C is the number
      of image channels, W is the width, and H is the height.
    batch_files : List[str]
      The filenames in the batch
    """
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        images = [transform(Image.open(f).convert("RGB")) for f in batch_files]
        yield torch.stack(images), batch_files
