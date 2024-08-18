import gradio as gr
import os
import requests
import numpy as np
import json
import sys
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor
from typing import List


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
    Initialize the connection to the PostgreSQL database with pgvector extension.

    Parameters
    ----------
    config : dict
        PostgreSQL configuration settings.
    """
    global conn, cursor
    conn = psycopg2.connect(
        host=config["postgresql"]["host"],
        port=config["postgresql"]["port"],
        database=config["postgresql"]["database"],
        user=config["postgresql"]["user"],
        password=config["postgresql"]["password"],
    )
    cursor = conn.cursor()

    # Create table for storing image embeddings if it doesn't exist
    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS image_embeddings (
        id SERIAL PRIMARY KEY,
        image_path TEXT UNIQUE,
        embedding VECTOR({config['vector_dimension']})
    );
    """
    )
    conn.commit()


def clear_database():
    """
    Clear the image_embeddings table in the database.
    """
    cursor.execute("DELETE FROM image_embeddings;")
    conn.commit()


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
        images = [open(image, "rb").read() for image in batch]

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

    with ThreadPoolExecutor() as executor:
        embeddings = list(
            executor.map(
                lambda paths: get_embeddings(paths, mlserver_endpoint), [image_paths]
            )
        )

    # Store embeddings in PostgreSQL
    data = [(path, embedding) for path, embedding in zip(image_paths, embeddings)]
    execute_values(
        cursor,
        "INSERT INTO image_embeddings (image_path, embedding) VALUES %s ON CONFLICT (image_path) DO NOTHING",
        data,
    )
    conn.commit()

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
    cursor.execute(
        "SELECT embedding FROM image_embeddings WHERE image_path = %s", (image_path,)
    )
    query_embedding = cursor.fetchone()[0]

    cursor.execute(
        """
    SELECT image_path
    FROM image_embeddings
    ORDER BY embedding <-> %s
    LIMIT 10;
    """,
        (query_embedding,),
    )

    similar_image_paths = [row[0] for row in cursor.fetchall()]
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
            image_input = gr.File(type="file", file_count="directory")
            gallery = gr.Gallery()
            process_button = gr.Button("Process")

        with gr.Row():
            similar_images_gallery = gr.Gallery()

        process_button.click(
            fn=lambda images: process_images(images, config["mlserver"]["endpoint"]),
            inputs=image_input,
            outputs=gallery,
        )
        gallery.select(
            fn=display_similar_images, inputs=gallery, outputs=similar_images_gallery
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
