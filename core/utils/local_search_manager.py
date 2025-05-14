# core/utils/local_search_manager.py

import os
import json
import numpy as np
import traceback

from langchain_community.embeddings import HuggingFaceEmbeddings

class LocalSearchManager:
    """
    Pure-Python fallback vector store using NumPy for embedding storage
    and cosine-similarity search, with JSON persistence.
    """
    def __init__(self, persist_path: str = "./vs_store.json"):
        self.persist_path = persist_path
        # initialize embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        # determine embedding dimension by querying an empty string
        try:
            sample_emb = self.embedder.embed_query("")
            dim = len(sample_emb)
        except Exception:
            dim = 384  # fallback dimension
        
        # load existing store or initialize empty arrays
        if os.path.exists(self.persist_path):
            try:
                data = json.load(open(self.persist_path, "r", encoding="utf-8"))
                self.docs = data.get("docs", [])
                embs_list = data.get("embs", [])
                self.embs = np.array(embs_list)
            except Exception:
                self.docs = []
                self.embs = np.empty((0, dim))
        else:
            self.docs = []
            self.embs = np.empty((0, dim))

    def import_document(self, file_path: str) -> str:
        """
        Load, split, embed, and index file chunks using pure-Python store.
        Requires implementing your own _load_and_split function.
        """
        from core.utils.local_search_manager_fallback_loader import _load_and_split

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # load and split
            chunks = _load_and_split(file_path)
            if not chunks:
                raise RuntimeError("No chunks generated from the file.")

            # embed texts
            texts = [chunk.page_content for chunk in chunks]
            new_embs = np.array(self.embedder.embed_documents(texts))

            # add to store
            for idx, chunk in enumerate(chunks):
                self.docs.append({
                    "source": chunk.metadata.get("source", ""),
                    "page_content": chunk.page_content
                })
            if new_embs.ndim == 1:
                new_embs = new_embs.reshape(1, -1)
            self.embs = np.vstack([self.embs, new_embs]) if self.embs.size else new_embs

            # persist
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "docs": self.docs,
                    "embs": self.embs.tolist()
                }, f)
            return file_path
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Import failed: {e}") from e

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Perform cosine-similarity search over stored embeddings.
        """
        try:
            if len(self.docs) == 0:
                return "⚠️ No documents indexed yet. Please import a file first."

            q_emb = np.array(self.embedder.embed_query(query))
            # normalize
            emb_norms = np.linalg.norm(self.embs, axis=1)
            q_norm = np.linalg.norm(q_emb)
            sims = (self.embs @ q_emb) / (emb_norms * q_norm + 1e-10)
            idxs = np.argsort(sims)[-top_k:][::-1]

            results = []
            for i in idxs:
                doc = self.docs[i]
                results.append(f"🔍 {doc['source']}\n{doc['page_content']}")
            return "\n\n".join(results)
        except Exception as e:
            print(f"[LocalSearchManager] Search error: {e}")
            return "⚠️ Failed to perform document search."
