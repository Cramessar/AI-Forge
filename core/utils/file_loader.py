# file_loader.py — Flexible loader for various document formats and standardized ingestion

import os
import shutil
import fitz  # PyMuPDF
import docx
import csv
import pandas as pd
from xml.etree import ElementTree
from bs4 import BeautifulSoup

SUPPORTED_EXTENSIONS = [".txt", ".md", ".log", ".pdf", ".docx", ".csv", ".xls", ".xlsx", ".xml"]

# All user-imported files will be copied here
DOCS_DIR = os.path.abspath("plugins/local_search/docs")

def is_supported(filepath: str) -> bool:
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def ingest_file(filepath: str) -> str:
    """
    Copies a supported file to the standard ingest directory and returns the new absolute path.
    """
    if not is_supported(filepath):
        raise ValueError(f"Unsupported file type: {filepath}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    filename = os.path.basename(filepath)
    destination = os.path.join(DOCS_DIR, filename)
    shutil.copy2(filepath, destination)
    return os.path.abspath(destination)

def load_file(filepath: str) -> str:
    """
    Loads and returns the text content of a file. Supports PDF, DOCX, TXT, CSV, XML, etc.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".txt", ".md", ".log"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            text = ""
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
            return text

        elif ext == ".docx":
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])

        elif ext in [".csv", ".xls", ".xlsx"]:
            df = pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)
            return df.to_string(index=False)

        elif ext == ".xml":
            tree = ElementTree.parse(filepath)
            root = tree.getroot()
            return ElementTree.tostring(root, encoding="unicode", method="text")

    except Exception as e:
        print(f"[FileLoader] Failed to load {filepath}: {e}")

    return ""
