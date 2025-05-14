# core/utils/local_search_manager_fallback_loader.py

import os
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document


def _load_and_split(file_path: str) -> list[Document]:
    """
    Load a file into LangChain Documents based on its extension,
    then split those documents into smaller chunks for indexing.
    Returns a list of Document objects with metadata['source'] set.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # 1) Load raw documents based on file type
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
    elif ext in [".csv", ".tsv"]:
        loader = CSVLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        # txt, md, markdown, log, xml, etc.
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

    # 2) Split each Document into chunks
    out_chunks: list[Document] = []
    for doc in docs:
        # ensure metadata exists and set source
        source = doc.metadata.get('source') if hasattr(doc, 'metadata') else None
        source = source or os.path.basename(file_path)

        # choose splitter by extension
        if ext in [".md", ".markdown"]:
            splitter = MarkdownHeaderTextSplitter(chunk_size=500, chunk_overlap=50)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

        # split_documents returns a list of Documents
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            # set chunk metadata source
            chunk.metadata = getattr(chunk, 'metadata', {}) or {}
            chunk.metadata['source'] = source
            out_chunks.append(chunk)

    return out_chunks
