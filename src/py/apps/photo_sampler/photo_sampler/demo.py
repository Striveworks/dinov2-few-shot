import gradio as gr
import numpy as np
import json
import sys
import torch

from sqlalchemy import (
    create_engine,
    delete,
    text,
)

from sqlalchemy.orm import sessionmaker
from transformers import AutoImageProcessor, AutoModel

from typing import List
from PIL import Image
from tqdm import tqdm

from photo_sampler.model import ImageEmbedding, Base
from photo_sampler.cluster import cluster_embeddings, select_sample


def load_config(config_path: str) -> dict:
    """
    Load the configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    dict
        Configuration settings loaded from the JSON file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def initialize_pgvector(config: dict):
    """
    Initialize the connection to the PostgreSQL database with SQLAlchemy.

    Parameters
    ----------
    config : dict
        PostgreSQL configuration settings.
    """
    global Session, engine

    db_url = f"postgresql+psycopg2://{config['postgresql']['user']}:{config['postgresql']['password']}@{config['postgresql']['host']}:{config['postgresql']['port']}/{config['postgresql']['database']}"
    engine = create_engine(db_url)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Clear existing data
    clear_database(hard_delete=config.get("hard_delete") or True)

    # Create the table for storing image embeddings if it doesn't exist
    Base.metadata.create_all(engine)

    # Create the vector extension
    with Session() as session:
        with session.begin():
            session.execute(text("CREATE EXTENSION IF NOT EXISTS VECTOR;"))


def clear_database(hard_delete: bool = True):
    """
    Clear the image_embeddings table in the database.

    Parameters
    ----------
    hard_delete : bool
      If true, tables will be dropped instead of just entries.
    """
    session = Session()
    if hard_delete:
        Base.metadata.drop_all(engine)
    else:
        session.query(ImageEmbedding).delete()
    session.commit()
    session.close()


def reset_application():
    """Reset the entire application state."""
    session = Session()

    # Clear the database
    session.execute(delete(ImageEmbedding))
    session.commit()
    session.close()

    # Return empty outputs for galleries
    return (
        [],
        [],
        0.5,
    )  # Return an empty list for both galleries and reset granularity slider


def reprocess_gallery(image_files, granularity):
    """Rerun the update_gallery function after resetting the database."""
    reset_application()
    return update_gallery(image_files, granularity)


def process_images(
    image_files: list, n_clusters: int = 10, pr=gr.Progress(track_tqdm=True)
) -> tuple:
    """Process images to extract embeddings and store them in the database.

    Parameters
    ----------
    image_files : list
        List of uploaded image files.
    n_clusters : int
        The number of clusters to use for moments

    Returns
    -------
    tuple
        Tuple containing the list of processed images and their embeddings.
    """
    session = Session()

    # Clear the database
    session.execute(delete(ImageEmbedding))
    session.commit()

    # Get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(device)
    embeddings = []
    images = []

    # for image_file in tqdm(image_files, desc="Processing Images"):
    for image_file in pr.tqdm(image_files, desc="Processing Images"):
        image = Image.open(image_file)
        images.append(image)

        # Preprocess image and get embedding
        inputs = processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        with torch.no_grad():
            embedding = (
                model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            )
        embeddings.append(embedding)

        # Store vectors
        image_embedding = ImageEmbedding(
            image_path=image_file.name,
            embedding=embedding.tolist(),
        )
        session.add(image_embedding)

    session.commit()
    session.close()

    # Do the clustering
    clusters = cluster_embeddings(embeddings, n_clusters=n_clusters)
    return images, embeddings, clusters


# def display_similar_images(image_paths: List[str]) -> List[str]:
def display_similar_images(selection: gr.SelectData) -> List[str]:
    """
    Display the top 10 most similar images from PostgreSQL based on the selected image.

    Parameters
    ----------
    selection : gr.SelectData
        The selection determined by the gradio state at the time
        this function is called.

    Returns
    -------
    List[str]
        A list of paths to the top 10 similar images.
    """
    session = Session()
    image_path = selection.value["image"]["path"]

    query_embedding = (
        session.query(ImageEmbedding).filter_by(image_path=image_path).first().embedding
    )
    # Use the SQLAlchemy l2 vector similarity query (assuming pgvector extension)
    similar_images = (
        session.query(
            ImageEmbedding,
            ImageEmbedding.embedding.l2_distance(query_embedding).label("distance"),
        )
        .order_by("distance")
        .limit(10)
        .all()
    )
    similar_image_paths = [x[0].image_path for x in similar_images]
    session.close()

    return similar_image_paths


def update_gallery(image_input: List[gr.File], granularity: float) -> List[str]:
    """
    Update the gallery with the paths of uploaded images.

    Parameters
    ----------
    image_input : List[gr.File]
        A list of uploaded images.

    Returns
    -------
    List[str]
        A list of paths to the uploaded images.
    """
    image_paths = [img.name for img in image_input]
    _, _, clusters = process_images(
        image_input, n_clusters=int(granularity * len(image_paths))
    )
    sample = select_sample(image_paths, clusters)
    return sample  # image_paths


if __name__ == "__main__":
    # Load configuration from JSON file passed via command line
    if len(sys.argv) != 2:
        print("Usage: python deno.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    # Initialize PostgreSQL with pgvector
    initialize_pgvector(config)

    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("# Image Exploration Application")
        gr.Markdown(
            "Use AI to explore the unique points in your image gallery and find high quality photos."
        )
        gr.Markdown("## Usage")
        gr.Markdown(
            """
            Simply upload a folder with your images. Resize beforehand for best results. After upload, the 
            images will be processed and unique moments will be entered into the gallery on the right. Select
            an image to populate the bottom gallery with similar images. To reset the application, simply 
            press reset on the bottom row and clear the upload box. To re-run with the same uploaded images
            and a different granularity, press the reprocess button after resetting the application.
            """
        )
        gr.Markdown("## Granularity Slider")
        gr.Markdown("The ratio of unique moments to total images.")
        with gr.Row():
            granularity_slider = gr.Slider(
                minimum=0, maximum=1, step=0.01, value=0.5, label="Granularity"
            )

        with gr.Row():
            image_input = gr.Files(
                type="filepath",
                file_count="directory",
                label="Upload Files",
                height=512,
            )
            gallery = gr.Gallery(label="Cluster Centers")

        with gr.Row():
            similar_images_gallery = gr.Gallery()

        with gr.Row():
            reset_button = gr.Button("Reset")
        with gr.Row():
            reprocess_button = gr.Button("Reprocess")

        image_input.upload(
            fn=update_gallery,
            inputs=[image_input, granularity_slider],
            outputs=gallery,
            show_progress="full",
            concurrency_limit=10,
        )

        reprocess_button.click(
            fn=reprocess_gallery,
            inputs=[image_input, granularity_slider],
            outputs=gallery,
            show_progress=True,
        )

        gallery.select(
            fn=display_similar_images, inputs=None, outputs=similar_images_gallery
        )

        reset_button.click(
            fn=reset_application,
            inputs=None,
            outputs=[gallery, similar_images_gallery, granularity_slider],
        )
    # Set share to False if running containerized
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
