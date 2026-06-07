"""Orchestration de l'enrichissement à la volée du dataset chargé.

Trois calculs indépendants, déclenchés par 3 boutons distincts dans l'UI :
- recatégorisation Nature  -> Nature_predite, Nature_Score, Nature_Commentaire
- extraction couleur       -> couleur_extraite, Couleur_Commentaire, ...
- extraction dimension     -> dim_label, Dimension_Commentaire, ...

Le « dataset de travail » est conservé en session (st.session_state) et chaque
bouton vient y AJOUTER ses colonnes. Il est réinitialisé automatiquement quand on
change de fichier source. Si un livrable enrichi est déjà chargé comme fichier
source, ses colonnes sont simplement détectées comme déjà présentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.data_loader import obtenir_dataframe_actif, reinitialiser_si_changement_dataset
from core.extract import extraire_couleur_contextuelle_serie, extraire_dimensions_v2_serie
from core.recat import executer_recat

WORK_KEY = "work_enriched_df"
SRC_KEY = "work_enriched_src"

RECAT_COLS = ["Nature_predite", "Nature_Score", "Nature_Commentaire",
              "Nature_Score_Pass1", "Nature_Score_Pass2"]
COLOR_COLS = ["couleur_extraite", "couleurs_toutes", "nb_couleurs_detectees",
              "couleur_decision", "couleur_score", "Couleur_Commentaire"]
DIM_COLS = ["L_cm", "l_cm", "H_cm", "dim_label", "diametre_cm", "dimension_simple_cm",
            "dimension_simple_type", "dimension_source", "dimension_warning",
            "dimension", "dimension_alerte", "nb_dimensions", "Dimension_Commentaire"]


# ------------------------------------------------------------------
# Gestion du dataset de travail (session)
# ------------------------------------------------------------------
def obtenir_df_travail() -> pd.DataFrame | None:
    """Dataset de travail courant (= fichier chargé + enrichissements déjà calculés).

    Réinitialisé si le fichier source a changé.
    """
    reinitialiser_si_changement_dataset()
    src = st.session_state.get("dataset_path")
    if src is None:
        return None
    if st.session_state.get(SRC_KEY) != src or WORK_KEY not in st.session_state:
        base = obtenir_dataframe_actif()
        if base is None:
            return None
        st.session_state[WORK_KEY] = base.copy()
        st.session_state[SRC_KEY] = src
    return st.session_state[WORK_KEY]


def _store(df: pd.DataFrame) -> None:
    st.session_state[WORK_KEY] = df


def possede_colonnes(cols: list[str]) -> bool:
    df = st.session_state.get(WORK_KEY)
    return df is not None and all(c in df.columns for c in cols)


def reinitialiser_df_travail() -> None:
    st.session_state.pop(WORK_KEY, None)
    st.session_state.pop(SRC_KEY, None)


# ------------------------------------------------------------------
# 1) Recatégorisation Nature
# ------------------------------------------------------------------
def calculer_recat(two_pass: bool = False, seuil_p2: float = 0.50, seuil_p1: float = 0.80,
              use_vendeur: bool = True, use_prix: bool = True) -> dict:
    df = obtenir_df_travail()
    if df is None:
        return {"ok": False, "msg": "Aucun dataset chargé."}
    # On repart toujours du dataset de base pour la recat (sinon Pass2 réécrirait par-dessus
    # une Nature déjà modifiée). On retire d'éventuelles colonnes recat précédentes.
    base = df.drop(columns=[c for c in RECAT_COLS if c in df.columns], errors="ignore")
    out, info = executer_recat(base, two_pass=two_pass, seuil_p2=seuil_p2, seuil_p1=seuil_p1,
                          use_vendeur=use_vendeur, use_prix=use_prix)
    if info.get("ok"):
        _store(out)
        st.session_state["recat_info"] = info  # purgé au changement de dataset (préfixe "recat_")
    return info


# ------------------------------------------------------------------
# 2) Extraction couleur
# ------------------------------------------------------------------
def calculer_couleur(fine: bool = True) -> dict:
    df = obtenir_df_travail()
    if df is None or "Libelle" not in df.columns:
        return {"ok": False, "msg": "Colonne 'Libelle' absente."}
    df = df.copy()
    s = df["Libelle"].astype("string")
    uniq = pd.Series(s.dropna().unique())
    res = extraire_couleur_contextuelle_serie(uniq, fine=fine)
    res.index = uniq.values

    df["couleur_extraite"] = s.map(res["couleur_niveau3"])
    df["couleurs_toutes"] = s.map(res["couleurs_detectees"])
    nb = s.map(res["nb_couleurs_detectees"]).fillna(0).astype(int)
    df["nb_couleurs_detectees"] = nb
    df["couleur_decision"] = s.map(res["decision_niveau3"])
    df["couleur_score"] = s.map(res["score_niveau3"])
    df["Couleur_Commentaire"] = np.select(
        [nb.values == 0, nb.values == 1],
        ["Aucune couleur", "Couleur unique"],
        default=nb.astype(str).values + " couleurs",
    )
    _store(df)
    return {
        "ok": True,
        "n_avec_couleur": int(df["couleur_extraite"].notna().sum()),
        "n_multi": int((nb >= 2).sum()),
        "n_total": int(len(df)),
    }


# ------------------------------------------------------------------
# 3) Extraction dimension
# ------------------------------------------------------------------
_ALERTE_FR = {
    "decimal_space_repaired": "décimale réparée",
    "unite_m_suspecte": "unité (m) suspecte",
    "valeur_suspecte": "valeur suspecte",
    "rejet_puissance": "ignoré : puissance (W)",
    "rejet_reference": "ignoré : référence produit",
    "rejet_resolution": "ignoré : résolution écran",
}


def _fmt_dim(v) -> str | None:
    return f"{float(v):g}" if pd.notna(v) else None


def _dim_unifie(r) -> str | None:
    s = r["dimension_source"]
    if s in ("regex_standard", "regex_normalisee_decimal_espace") and pd.notna(r["dim_label"]):
        return f"{r['dim_label']} cm"
    if s == "diametre" and pd.notna(r["diametre_cm"]):
        lab = "Ø " + _fmt_dim(r["diametre_cm"])
        if pd.notna(r["H_cm"]):
            lab += "xH" + _fmt_dim(r["H_cm"])
        return lab + " cm"
    if s == "simple_cm" and pd.notna(r["dimension_simple_cm"]):
        return f"{_fmt_dim(r['dimension_simple_cm'])} cm"
    return None


def _dim_nb(r) -> int:
    s = r["dimension_source"]
    if s in ("regex_standard", "regex_normalisee_decimal_espace"):
        return int(pd.notna(r["L_cm"]) + pd.notna(r["l_cm"]) + pd.notna(r["H_cm"]))
    if s == "diametre":
        return int(pd.notna(r["diametre_cm"]) + pd.notna(r["H_cm"]))
    if s == "simple_cm" and pd.notna(r["dimension_simple_cm"]):
        return 1
    return 0


def calculer_dimension() -> dict:
    df = obtenir_df_travail()
    if df is None or "Libelle" not in df.columns:
        return {"ok": False, "msg": "Colonne 'Libelle' absente."}
    df = df.copy()
    s = df["Libelle"].astype("string")
    uniq = pd.Series(s.dropna().unique())
    dim_u = extraire_dimensions_v2_serie(uniq)
    dim_u.index = uniq.values
    dim_u["dimension"] = dim_u.apply(_dim_unifie, axis=1)
    dim_u["dimension_alerte"] = dim_u["dimension_warning"].map(_ALERTE_FR)
    dim_u["nb_dimensions"] = dim_u.apply(_dim_nb, axis=1)

    for c in ["L_cm", "l_cm", "H_cm", "dim_label", "diametre_cm", "dimension_simple_cm",
              "dimension_simple_type", "dimension_source", "dimension_warning",
              "dimension", "dimension_alerte", "nb_dimensions"]:
        df[c] = s.map(dim_u[c])

    nb_axes = df[["L_cm", "l_cm", "H_cm"]].notna().sum(axis=1).values
    has_diam = df["diametre_cm"].notna().values
    has_simple = df["dimension_simple_cm"].notna().values
    has_warning = df["dimension_warning"].notna().values
    df["Dimension_Commentaire"] = np.select(
        [
            (nb_axes == 0) & ~has_diam & ~has_simple,
            has_warning & (nb_axes > 0),
            nb_axes == 3,
            nb_axes == 2,
            has_diam,
            nb_axes == 1,
            has_simple,
        ],
        ["Aucune dimension", "Avec alerte", "3 dimensions", "2 dimensions",
         "Diamètre", "1 dimension", "1 dimension"],
        default="Aucune dimension",
    )
    _store(df)
    return {
        "ok": True,
        "n_avec_dim": int((df["Dimension_Commentaire"] != "Aucune dimension").sum()),
        "n_diam": int(has_diam.sum()),
        "n_total": int(len(df)),
    }
