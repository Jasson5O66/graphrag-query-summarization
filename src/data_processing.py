import os
import requests
import logging
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_graphrag.indexing import TextUnitExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional

def download_pdf(url: str, save_path: str):
    """
    Downloads a PDF file from a URL.

    Args:
        url (str): The URL of the PDF file.
        save_path (str): The local path to save the file.
    """
    logging.info(f"Downloading PDF from {url} to {save_path}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logging.info(f"Successfully downloaded PDF: {save_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download PDF: {e}", exc_info=True)
        raise

def load_and_split_docs(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a PDF and splits it into text units using LangChain components.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing text units, or None if failed.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    logging.info(f"Loading documents from {file_path}...")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} pages from PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        text_unit_extractor = TextUnitExtractor(text_splitter=splitter)

        logging.info("Extracting text units (chunks)...")
        # The run method expects a list of Document objects
        df_text_units = text_unit_extractor.run(docs)
        
        logging.info(f"Extracted {len(df_text_units)} text units.")
        return df_text_units

    except Exception as e:
        logging.error(f"Failed to load or split documents: {e}", exc_info=True)
        return None
