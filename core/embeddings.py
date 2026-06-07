"""Embeddings sémantiques + projection 2D/3D (UMAP).

Tout est optionnel : si sentence-transformers ou umap ne sont pas installés,
on retombe sur TF-IDF + TruncatedSVD.
"""
from __future__ import annotations
import os
import importlib.util
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
HAS_ST = (
    os.getenv("APP_USE_SENTENCE_TRANSFORMERS") == "1"
    and importlib.util.find_spec("sentence_transformers") is not None
)
HAS_UMAP = (
    os.getenv("APP_USE_UMAP") == "1"
    and importlib.util.find_spec("umap") is not None
)


def _encode_tfidf_svd(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
    X = vec.fit_transform(texts)
    if X.shape[1] < 2:
        return X.toarray()
    n_components = min(128, max(1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)


def _pad_projection(arr: np.ndarray, n_components: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] >= n_components:
        return arr[:, :n_components]
    pad = np.zeros((arr.shape[0], n_components - arr.shape[1]))
    return np.hstack([arr, pad])


def encoder_textes(texts: list[str], cache_path: str | None = None) -> np.ndarray:
    """Encode des textes en vecteurs denses. Utilise sentence-transformers si dispo,
    sinon TF-IDF + SVD."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    if HAS_ST:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
            emb = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        except Exception:
            emb = _encode_tfidf_svd(texts)
    else:
        emb = _encode_tfidf_svd(texts)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, emb)
    return emb


def reduire_2d(emb: np.ndarray, random_state: int = 42) -> np.ndarray:
    if HAS_UMAP:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(emb)
        except Exception:
            pass
    if emb.shape[1] < 2:
        return _pad_projection(emb, 2)
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    return svd.fit_transform(emb)


def reduire_3d(emb: np.ndarray, random_state: int = 42) -> np.ndarray:
    if HAS_UMAP:
        try:
            import umap

            reducer = umap.UMAP(n_components=3, random_state=random_state, n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(emb)
        except Exception:
            pass
    if emb.shape[1] < 3:
        return _pad_projection(emb, 3)
    svd = TruncatedSVD(n_components=3, random_state=random_state)
    return svd.fit_transform(emb)


def echantillon_stratifie(df: pd.DataFrame, n: int, by: str = "Nature", random_state: int = 42) -> pd.DataFrame:
    """Échantillonne ~n lignes en gardant la stratification par 'by'."""
    if len(df) <= n:
        return df
    if by not in df.columns:
        return df.sample(n=n, random_state=random_state)
    g = df.groupby(by, group_keys=False)
    return g.apply(lambda x: x.sample(min(len(x), max(1, n * len(x) // len(df))), random_state=random_state))
