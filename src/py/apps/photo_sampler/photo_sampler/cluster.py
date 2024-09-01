import numpy as np

from sklearn.cluster import KMeans
from photo_sampler.model import ImageEmbedding
from typing import List


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int = 10, n_init: int = 10
) -> List[int]:
    """
    Parameters
    ----------
    embeddings : np.ndarray
        The stored embeddings
    n_clusters : int
        The number of clusters to use
    n_init : int
        The number of KMeans trials

    Returns
    -------
    cluster_labels : List[int]
        The list of cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    clusters = kmeans.fit_predict(embeddings)
    return clusters


def select_sample(files: List[str], labels: np.ndarray, n_per_index: int = 1):
    """
    files : List[str]
        The list of filepaths
    labels : np.ndarray
        The labels for each file
    n_per_index: int
        Number of samples for each index
    """
    file_sample = []
    file_array = np.array(files)
    for c in np.unique(labels):
        sample = np.random.choice(file_array[labels == c], replace=False)
        file_sample += [sample.tolist()]
    return file_sample
