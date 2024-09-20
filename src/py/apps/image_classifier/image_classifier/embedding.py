import torch

from PIL import Image
from image_classifier.encoder import FaissIndexImageEncoder
from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor, ViTModel


def mean_aggregate(x):
    embeddings = x.last_hidden_state
    return embeddings.mean(axis=1)


def get_embeddings_and_labels(image_folder: str):
    """
    Get the embeddings for the classifier.

    Parameters
    ----------
    image_folder : str
        A folder structured as folder/class_name/image_x.png

    Returns
    -------
    embeddings : np.ndarray
        The embeddings for the images
    labels : List[str]
        The labels for the classifier
    label_idxs : np.ndarray
        The integer mapped labels
    image_paths : List[str]
        The paths to the images at each index
    """
    # Initialize model on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First embedding model: dinov2
    processor_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    processor_kwargs_dino = {"return_tensors": "pt"}

    dinov2_hf = AutoModel.from_pretrained("facebook/dinov2-large")

    dinov2 = (
        FaissIndexImageEncoder(
            dinov2_hf,
            index_file_out="./data/test_photos_dinov2.db",
        )
        .eval()
        .to(device)
    )

    # Walk over the photo directory
    with torch.no_grad():
        image_paths = []
        labels = []
        class_names = os.listdir(image_folder)

        label_map = {c: i for i, c in enumerate(class_names)}
        int_map = {i: c for i, c in enumerate(class_names)}

        for class_name in class_names:
            class_folder = os.path.join(image_folder, class_name)
            if os.path.isdir(class_folder):
                for image_name in tqdm(os.listdir(class_folder)):
                    x = Image.open(os.path.join(class_folder, image_name))
                    y = processor_dino(images=x, return_tensors="pt")
                    # pixel_values = torch.stack(tuple([y["pixel_values"][0] for y in x]))
                    dino_batch = {"pixel_values": y["pixel_values"].to(device)}
                    # dinov2_embeddings = dinov2(dino_batch, files, mean_aggregate)
                    dinov2_embeddings = dinov2(
                        dino_batch,
                        [os.path.join(class_folder, image_name)],
                        mean_aggregate,
                    )
                    image_paths.append(os.path.join(class_folder, image_name))
                    labels.append(class_name)
        label_idxs = np.array([label_map[l] for l in labels])
        return dinov2.current_embeddings, labels, label_idxs, image_paths
