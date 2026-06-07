"""Utilitaires de cache disque pour les résultats de calculs lourds (déterministes).

Idée : on calcule une clé unique à partir du dataset, des paramètres et de la
version du code, puis on stocke/relit le résultat dans `data/cache/*.pkl`.
Cela évite de recalculer l'index de recherche (page 0) ou le graphe / la
projection / le Sankey (page 2) à chaque visite de page.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def repertoire_cache() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def hash_fichier(path: str | os.PathLike[str]) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return "no-file-hash"


def hash_commit_git() -> str:
    """Stable cache namespace.

    Historiquement, les cles de cache incluaient le hash Git HEAD. C'etait trop
    agressif : un commit documentaire ou UI invalidait tous les calculs lourds.
    On conserve le champ `git` dans le payload pour compatibilite de structure,
    mais sa valeur est maintenant un namespace persistant. Les invalidations
    utiles restent pilotees par `version`, la signature du dataframe, les
    parametres et les hash des fichiers de code passes en `code_files`.
    """
    app_dir = os.path.dirname(os.path.dirname(__file__))
    repo = os.path.dirname(app_dir)
    namespace_path = os.path.join(app_dir, "data", "cache", "cache_namespace.txt")
    if os.path.exists(namespace_path):
        try:
            with open(namespace_path, "r", encoding="utf-8") as f:
                value = f.read().strip()
            if value:
                return value
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        value = result.stdout.strip() or "no-git-head"
    except Exception:
        value = "no-git-head"

    try:
        os.makedirs(os.path.dirname(namespace_path), exist_ok=True)
        with open(namespace_path, "w", encoding="utf-8") as f:
            f.write(value)
    except Exception:
        pass
    return value


def signature_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "empty"
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not cols:
        return f"rows={len(df)}|no-cols"
    work = df[cols].copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].astype("datetime64[ns]").astype("int64")
        else:
            work[col] = work[col].astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(work, index=True).values
    digest = hashlib.md5(row_hashes.tobytes()).hexdigest()[:16]
    return f"rows={len(df)}|cols={','.join(cols)}|hash={digest}"


def construire_cle_cache(
    kind: str,
    version: str,
    df: pd.DataFrame,
    columns: list[str] | None = None,
    params: dict | None = None,
    code_files: list[str | os.PathLike[str]] | None = None,
) -> str:
    payload = {
        "kind": kind,
        "version": version,
        "data": signature_dataframe(df, columns=columns),
        "params": params or {},
        "git": hash_commit_git(),
        "code": {str(Path(p)): hash_fichier(p) for p in (code_files or [])},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:18]


def chemin_cache_pickle(kind: str, key: str) -> str:
    safe_kind = "".join(c if c.isalnum() or c in "-_" else "_" for c in kind)
    return os.path.join(repertoire_cache(), f"{safe_kind}_{key}.pkl")


def charger_cache_pickle(kind: str, key: str):
    path = chemin_cache_pickle(kind, key)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def sauvegarder_cache_pickle(kind: str, key: str, data) -> None:
    path = chemin_cache_pickle(kind, key)
    try:
        pd.to_pickle(data, path)
    except Exception:
        pass
