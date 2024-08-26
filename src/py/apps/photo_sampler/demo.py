import gradio as gr
import os
import requests
import numpy as np
import json
import sys
import base64

from sqlalchemy import (
    create_engine,
    Column,
    Float,
    Integer,
    String,
    ARRAY,
    delete,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor
from typing import List
from PIL import Image

Base = declarative_base()


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id = Column(Integer, primary_key=True)
    image_path = Column(String, unique=True, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)


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

    # Create the table for storing image embeddings if it doesn't exist
    Base.metadata.create_all(engine)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create the vector extension
    with Session() as session:
        with session.begin():
            session.execute(text("CREATE EXTENSION IF NOT EXISTS VECTOR;"))


def clear_database():
    """
    Clear the image_embeddings table in the database.
    """
    session = Session()
    session.query(ImageEmbedding).delete()
    session.commit()
    session.close()


def get_embeddings(image_paths: List[str], mlserver_endpoint: str) -> List[np.ndarray]:
    """
    Get embeddings for a batch of images by sending them to the MLServer.

    Parameters
    ----------
    image_paths : List[str]
        A list of image file paths to be processed.
    mlserver_endpoint : str
        The MLServer endpoint for model inference.

    Returns
    -------
    List[np.ndarray]
        A list of embeddings for the provided images.
    """
    batch_size = 8
    embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        import pdb

        pdb.set_trace()

        # TODO: We need to apply the DinoV2 preprocessor. We should also resize images
        # for the gallery.

        images = [
            base64.b64encode(Image.open(image).tobytes()).decode() for image in batch
        ]

        response = requests.post(mlserver_endpoint, json={"instances": images})
        batch_embeddings = response.json()["predictions"]
        embeddings.extend(batch_embeddings)

    return embeddings


def process_images(images: List[gr.File], mlserver_endpoint: str) -> List[str]:
    """
    Process uploaded images to obtain embeddings, clear the database, and store them in PostgreSQL.

    Parameters
    ----------
    images : List[gr.File]
        A list of uploaded images.
    mlserver_endpoint : str
        The MLServer endpoint for model inference.

    Returns
    -------
    List[str]
        A list of paths to the processed images.
    """
    image_paths = [image.name for image in images]

    # Clear the database before inserting new vectors
    clear_database()

    with ThreadPoolExecutor(max_workers=4) as executor:
        embeddings = list(
            executor.map(
                lambda paths: get_embeddings(paths, mlserver_endpoint), [image_paths]
            )
        )

    # Store embeddings in PostgreSQL using SQLAlchemy
    session = Session()
    for path, embedding in zip(image_paths, embeddings):
        image_embedding = ImageEmbedding(image_path=path, embedding=embedding)
        session.add(image_embedding)
    session.commit()
    session.close()

    return image_paths


def display_similar_images(image_path: str) -> List[str]:
    """
    Display the top 10 most similar images from PostgreSQL based on the selected image.

    Parameters
    ----------
    image_path : str
        The path of the selected image.

    Returns
    -------
    List[str]
        A list of paths to the top 10 similar images.
    """
    session = Session()

    query_embedding = (
        session.query(ImageEmbedding).filter_by(image_path=image_path).first().embedding
    )

    # Use the SQLAlchemy vector similarity query (assuming pgvector extension)
    similar_images = session.execute(
        f"""
    SELECT image_path
    FROM image_embeddings
    ORDER BY embedding <-> ARRAY{query_embedding}
    LIMIT 10;
    """
    ).fetchall()

    session.close()

    return [row[0] for row in similar_images]


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
            fn=lambda images: process_images(images, config["mlserver"]["endpoint"]),
            inputs=image_input,
            outputs=gallery,
            show_progress=True,
        )
        gallery.select(
            fn=display_similar_images, inputs=gallery, outputs=similar_images_gallery
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
