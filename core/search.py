"""Recherche semantique legere sur les libelles produit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def construire_catalogue_libelles(df: pd.DataFrame) -> pd.DataFrame:
    """Agrege le dataset par libelle unique avec metadonnees majoritaires."""
    if "Libelle" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["Libelle"] = work["Libelle"].fillna("").astype(str).str.strip()
    work = work[work["Libelle"] != ""]
    if work.empty:
        return pd.DataFrame()

    aggregations = {"nb_lignes": ("Libelle", "size")}
    for col in ["Nature", "Univers", "Vendeur"]:
        if col in work.columns:
            aggregations[f"{col}_majoritaire"] = (col, _mode_or_none)

    catalog = work.groupby("Libelle", dropna=False).agg(**aggregations).reset_index()
    return catalog.sort_values("nb_lignes", ascending=False).reset_index(drop=True)


def _mode_or_none(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None
    return s.astype(str).value_counts().index[0]


def construire_texte_recherche(catalog: pd.DataFrame, mode: str = "Libelle seul") -> pd.Series:
    """Construit le texte indexe selon le mode de recherche."""
    text = catalog["Libelle"].fillna("").astype(str)
    if mode == "Libelle + Univers + Nature":
        parts = [text]
        for col in ["Univers_majoritaire", "Nature_majoritaire"]:
            if col in catalog.columns:
                parts.append(catalog[col].fillna("").astype(str))
        text = pd.Series([" ".join(values) for values in zip(*parts)], index=catalog.index)
    return text


def construire_index_tfidf(catalog: pd.DataFrame, mode: str = "Libelle seul"):
    """Cree un index TF-IDF word+char n-grams compatible CPU."""
    texts = construire_texte_recherche(catalog, mode=mode)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def rechercher_similaires(query: str, catalog: pd.DataFrame, vectorizer, matrix, top_k: int = 20) -> pd.DataFrame:
    """Retourne les top_k libelles les plus proches du texte query."""
    if not query or not query.strip() or catalog.empty:
        return pd.DataFrame()
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    if len(scores) == 0:
        return pd.DataFrame()
    top_k = min(top_k, len(scores))
    idx = np.argpartition(-scores, range(top_k))[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    out = catalog.iloc[idx].copy()
    out.insert(0, "score", scores[idx])
    return out.reset_index(drop=True)


def suggerer_categorie(neighbors: pd.DataFrame, top_n: int = 20) -> dict:
    """Suggere Nature/Univers a partir des voisins et calcule une confiance simple."""
    if neighbors.empty:
        return {
            "Nature_recommandee": None,
            "Univers_recommande": None,
            "confiance": "n/a",
            "score_moyen_top5": 0.0,
            "part_nature_top": 0.0,
        }

    sub = neighbors.head(top_n).copy()
    score_mean = float(sub["score"].head(5).mean()) if "score" in sub.columns else 0.0

    nature, nature_share = _weighted_top(sub, "Nature_majoritaire")
    univers, _ = _weighted_top(sub, "Univers_majoritaire")

    if score_mean >= 0.65 and nature_share >= 0.70:
        confiance = "elevee"
    elif score_mean >= 0.45 and nature_share >= 0.50:
        confiance = "moyenne"
    else:
        confiance = "faible"

    return {
        "Nature_recommandee": nature,
        "Univers_recommande": univers,
        "confiance": confiance,
        "score_moyen_top5": round(score_mean, 3),
        "part_nature_top": round(float(nature_share), 3),
    }


def _weighted_top(df: pd.DataFrame, col: str) -> tuple[str | None, float]:
    if col not in df.columns or df.empty:
        return None, 0.0
    tmp = df[[col, "score"]].dropna()
    if tmp.empty:
        return None, 0.0
    weighted = tmp.groupby(col)["score"].sum().sort_values(ascending=False)
    total = weighted.sum()
    if total <= 0:
        return str(weighted.index[0]), 0.0
    return str(weighted.index[0]), float(weighted.iloc[0] / total)
