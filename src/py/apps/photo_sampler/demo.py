import gradio as gr
import numpy as np
import json
import sys
import torch

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    delete,
    text,
)

from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import AutoImageProcessor, AutoModel

from typing import List
from PIL import Image
from tqdm import tqdm

Base = declarative_base()


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, unique=True, nullable=False)
    embedding = Column(Vector(1024))


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


def process_images(image_files: list) -> tuple:
    """Process images to extract embeddings and store them in the database.

    Parameters
    ----------
    image_files : list
        List of uploaded image files.

    Returns
    -------
    tuple
        Tuple containing the list of processed images and their embeddings.
    """
    session = Session()

    # Clear the database
    session.execute(delete(ImageEmbedding))
    session.commit()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").eval()
    embeddings = []
    images = []

    for image_file in tqdm(image_files, desc="Processing Images"):
        image = Image.open(image_file)
        images.append(image)

        # Preprocess image and get embedding
        inputs = processor(images=image, return_tensors="pt")
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

    return images, embeddings


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


def update_gallery(image_input: List[gr.File]) -> List[str]:
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
    return image_paths


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
        with gr.Row():
            image_input = gr.File(type="filepath", file_count="multiple", height=256)
            gallery = gr.Gallery()
            process_button = gr.Button("Process")

        with gr.Row():
            similar_images_gallery = gr.Gallery()

        upload_progress = gr.Progress(track_tqdm=True)
        image_input.upload(
            fn=update_gallery,
            inputs=[image_input],
            outputs=gallery,
            show_progress="full",
        )

        process_button.click(
            fn=lambda images: process_images(images),
            inputs=image_input,
            # outputs=gallery,
            show_progress=True,
        )

        # TODO: fix how we get the selected image
        # gallery.select(
        #     fn=display_similar_images, inputs=gallery, outputs=similar_images_gallery
        # )
        gallery.select(
            fn=display_similar_images, inputs=None, outputs=similar_images_gallery
        )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
