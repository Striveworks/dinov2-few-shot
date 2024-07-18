# photography-analysis
Some utilities for me to analyze photography. Example tasks include generating representative samples, filtering, and summary generation.

## Installation Instructions

This project uses Poetry for Python dependency management. Get the latest version of `poetry`. You can install it with pip inside a virtual environment like: 
```bash
pip install poetry
```
Once you have poetry, run:
```
poetry install
```
from the root level of the project.

## Project Structure

### Python Libraries

Reusable bits of Python code can be found in `src/py/libs`. Subdirectories are manually added to the `pyproject.toml` to form new packages.

### Python Applications

Things like FastAPI servers, dashboards, and other applications written in Python can be found in `src/py/apps`. Subdirectories are manually added to the `pyproject.toml` to form new packages.