"""Chargement + normalisation du dataset e-commerce."""
from __future__ import annotations
import hashlib
import os
import unicodedata
import pandas as pd
import streamlit as st


CANONICAL_COLS = [
    "Cod_cmd", "Libelle", "Vendeur", "Univers", "Nature",
    "Date_cmd", "Montant_cmd", "Quantite", "Prix_transport", "Delai_transport",
]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalize_colname(c: str) -> str:
    c = _strip_accents(str(c)).lower().strip()
    c = c.replace(" ", "_").replace("-", "_")
    return c


COL_ALIASES = {
    "cod_cmd": "Cod_cmd",
    "code_cmd": "Cod_cmd",
    "code_commande": "Cod_cmd",
    "libelle_produit": "Libelle",
    "libelle": "Libelle",
    "produit": "Libelle",
    "vendeur": "Vendeur",
    "univers": "Univers",
    "nature": "Nature",
    "date_de_commande": "Date_cmd",
    "date_commande": "Date_cmd",
    "date_cmd": "Date_cmd",
    "montant_cmd": "Montant_cmd",
    "montant": "Montant_cmd",
    "montant_commande": "Montant_cmd",
    "quantite": "Quantite",
    "qte": "Quantite",
    "prix_transport": "Prix_transport",
    "delai_transport_annonce": "Delai_transport",
    "delai_transport": "Delai_transport",
}


def _cache_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _file_signature(path: str) -> str:
    stat = os.stat(path)
    raw = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def _normalized_cache_path(path: str) -> str:
    return os.path.join(_cache_dir(), f"dataset_normalized_{_file_signature(path)}.parquet")


def chemin_cache_dataset_purifie(path: str | None = None) -> str | None:
    """Chemin disque du dataset purifie associe au fichier source actif."""
    source_path = path or st.session_state.get("dataset_path")
    if not source_path or not os.path.exists(source_path):
        return None
    return os.path.join(_cache_dir(), f"dataset_purifie_{_file_signature(source_path)}.parquet")


def sauvegarder_dataset_purifie(df: pd.DataFrame, path: str | None = None) -> str | None:
    """Persiste le dataset purifie pour le rendre disponible apres redemarrage."""
    cache_path = chemin_cache_dataset_purifie(path)
    if not cache_path:
        return None
    try:
        df.to_parquet(cache_path, index=False)
        return cache_path
    except Exception:
        return None


def charger_dataset_purifie(path: str | None = None) -> pd.DataFrame | None:
    """Recharge le dataset purifie persiste, si disponible."""
    cache_path = chemin_cache_dataset_purifie(path)
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def _read_source_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsb":
        return pd.read_excel(path, engine="pyxlsb", sheet_name=0)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=0)
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8", sep=None, engine="python")
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Format non supporte : {ext}")


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COL_ALIASES.get(_normalize_colname(c), c) for c in df.columns}
    df = df.rename(columns=rename)

    if "Date_cmd" in df.columns and pd.api.types.is_numeric_dtype(df["Date_cmd"]):
        df["Date"] = pd.to_datetime(df["Date_cmd"], origin="1899-12-30", unit="D", errors="coerce")
    elif "Date_cmd" in df.columns:
        df["Date"] = pd.to_datetime(df["Date_cmd"], errors="coerce")

    for c in ("Montant_cmd", "Quantite", "Prix_transport", "Delai_transport"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("Libelle", "Vendeur", "Univers", "Nature"):
        if c in df.columns:
            df[c] = pd.Series(
                [str(x).strip() if pd.notna(x) else None for x in df[c]],
                index=df.index,
                dtype=object,
            )

    if {"Montant_cmd", "Quantite"}.issubset(df.columns):
        df["CA"] = df["Montant_cmd"].fillna(0) * df["Quantite"].fillna(1)

    return df


@st.cache_data(show_spinner=False, ttl=3600)
def charger_dataset(path: str) -> pd.DataFrame:
    """Charge un fichier xlsb / xlsx / csv / parquet et normalise les colonnes.

    Un cache disque parquet est conserve entre redemarrages. Il est invalide
    automatiquement si le fichier source change de taille ou de date de modification.
    """
    ext = os.path.splitext(path)[1].lower()
    cache_path = _normalized_cache_path(path)
    if ext != ".parquet" and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = _normalize_dataset(_read_source_file(path))
    if ext != ".parquet":
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return df


def vider_cache_dataset(path: str | None = None) -> None:
    """Vide le cache Streamlit et, si possible, le parquet normalise du dataset."""
    st.cache_data.clear()
    if path:
        try:
            cache_path = _normalized_cache_path(path)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception:
            pass


def obtenir_dataframe_actif() -> pd.DataFrame | None:
    """Recupere le DataFrame actif depuis session_state."""
    path = st.session_state.get("dataset_path")
    if not path or not os.path.exists(path):
        return None
    return charger_dataset(path)


def appliquer_filtres_globaux(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les filtres sidebar globaux (vendeur, univers, date)."""
    if df is None or df.empty:
        return df
    f = st.session_state.get("filters", {})
    v = f.get("vendeurs") or []
    if len(v) > 0 and "Vendeur" in df.columns:
        df = df[df["Vendeur"].isin(v)]
    u = f.get("univers") or []
    if len(u) > 0 and "Univers" in df.columns:
        df = df[df["Univers"].isin(u)]
    dr = f.get("date_range")
    if dr is not None and "Date" in df.columns:
        if len(dr) == 2:
            d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
            df = df[(df["Date"] >= d0) & (df["Date"] <= d1)]
    return df


# Préfixes des clés de session "dérivées" du dataset : tout ce qui doit être
# purgé quand on charge un nouveau fichier (enrichissements, filtres, caches, exports).
_SEEN_KEY = "_active_dataset_path_seen"
_DERIVED_PREFIXES = (
    "work_enriched",    # dataset de travail + sa source (core.enrich)
    "filters",          # filtres globaux sidebar
    "export_xlsx",      # bytes Excel préparés (page Export)
    "viz_",             # options/sélections de la page graphe
    "_viz_options",     # caches d'options de la page graphe
    "cmp_",             # widgets du mode comparaison
    "mc_query",         # recherche multi-critères (page 0)
    "recat_",           # anciens résultats de recat éventuels
    "colors_",          # anciens caches couleur
    "dims_",            # anciens caches dimension
)


def reinitialiser_si_changement_dataset() -> bool:
    """Purge tout l'état dérivé si le fichier source a changé. Idempotent.

    À appeler en début de page (via afficher_filtres_sidebar et enrich.obtenir_df_travail).
    Retourne True si une réinitialisation a eu lieu.
    """
    path = st.session_state.get("dataset_path")
    if st.session_state.get(_SEEN_KEY) == path:
        return False
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(_DERIVED_PREFIXES):
            del st.session_state[k]
    st.session_state[_SEEN_KEY] = path
    return True


def afficher_filtres_sidebar(df: pd.DataFrame) -> None:
    """Sidebar de filtres globaux, persistes dans st.session_state['filters']."""
    if df is None or df.empty:
        return
    reinitialiser_si_changement_dataset()
    with st.sidebar:
        st.markdown("### Filtres globaux")
        f = st.session_state.setdefault("filters", {})
        vendeurs = sorted([str(v) for v in df["Vendeur"].dropna().unique()]) if "Vendeur" in df.columns else []
        univers = sorted([str(u) for u in df["Univers"].dropna().unique()]) if "Univers" in df.columns else []
        # On filtre les valeurs memorisees aux options du dataset courant : sinon, changer de
        # fichier ferait planter le widget (default hors de la liste d'options).
        _v_default = [v for v in f.get("vendeurs", []) if v in vendeurs]
        _u_default = [u for u in f.get("univers", []) if u in univers]
        f["vendeurs"] = st.multiselect("Vendeurs", vendeurs, default=_v_default)
        f["univers"] = st.multiselect("Univers", univers, default=_u_default)
        if "Date" in df.columns and df["Date"].notna().any():
            dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
            # Bornage de la periode memorisee aux dates du dataset courant.
            _prev = f.get("date_range")
            if isinstance(_prev, (list, tuple)) and len(_prev) == 2:
                lo = min(max(_prev[0], dmin), dmax)
                hi = min(max(_prev[1], dmin), dmax)
                _date_default = (lo, hi) if lo <= hi else (dmin, dmax)
            else:
                _date_default = (dmin, dmax)
            f["date_range"] = st.date_input(
                "Periode",
                value=_date_default,
                min_value=dmin,
                max_value=dmax,
            )
        if st.button("Reset filtres"):
            st.session_state["filters"] = {}
            st.rerun()


# ============================================================
# Dataset enrichi optionnel, déjà calculé et déposé dans <app>/data/livrables/.
# S'il est présent, on peut le charger directement comme source de données.
# ============================================================
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LIVRABLE_FINAL_PATH = os.path.normpath(
    os.path.join(_DATA_DIR, "livrables", "livrable_final.parquet")
)
PRED_PASS1_PATH = os.path.normpath(
    os.path.join(_DATA_DIR, "livrables", "v3g_predictions.parquet")
)
PRED_PASS2_PATH = os.path.normpath(
    os.path.join(_DATA_DIR, "livrables", "V3g_v2_sansUniv_predictions.parquet")
)


@st.cache_data(show_spinner=False, ttl=3600)
def charger_livrable_final() -> pd.DataFrame | None:
    """Charge le livrable final s'il existe (30 colonnes : original + Nature + Couleur + Dim)."""
    if not os.path.exists(LIVRABLE_FINAL_PATH):
        return None
    try:
        return pd.read_parquet(LIVRABLE_FINAL_PATH)
    except Exception:
        return None


def livrable_final_disponible() -> bool:
    return os.path.exists(LIVRABLE_FINAL_PATH)


def obtenir_dataset_avec_predictions() -> pd.DataFrame | None:
    """Retourne le livrable enrichi si dispo (avec Nature_predite, couleurs, dim).
    Sinon retourne le dataset brut. Tres rapide grace au cache parquet."""
    livrable = charger_livrable_final()
    if livrable is not None:
        return livrable
    return obtenir_dataframe_actif()
