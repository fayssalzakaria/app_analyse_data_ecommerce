"""Assistant IA (RAG) : aide à catégoriser et répond à des questions sur le dataset.

Principe RAG :
1. RÉCUPÉRATION — pour un libellé / une question, on retrouve les libellés les plus
   proches déjà catégorisés (réutilise l'index TF-IDF de core/search).
2. GÉNÉRATION — on envoie ces exemples à Gemini, qui suggère une Nature/Univers
   (existante ou nouvelle) ou rédige une réponse, en s'appuyant sur le contexte.

La clé est lue depuis st.secrets['GEMINI_API_KEY'] ou la variable d'env GEMINI_API_KEY.
Aucune donnée massive n'est envoyée au LLM : seulement un résumé + quelques exemples.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

from core.search import (
    construire_catalogue_libelles,
    construire_index_tfidf,
    rechercher_similaires,
)
from core.recat import colonne_predite

MODELE = "gemini-2.5-flash"


# ------------------------------------------------------------------
# Accès à la clé / au client Gemini
# ------------------------------------------------------------------
def cle_gemini() -> str | None:
    """Récupère la clé API depuis les secrets Streamlit ou l'environnement."""
    try:
        cle = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        cle = None
    return cle or os.environ.get("GEMINI_API_KEY")


def gemini_pret() -> bool:
    return bool(cle_gemini())


def _generer(prompt: str, json_mode: bool = False, temperature: float = 0.2, essais: int = 2) -> str:
    """Appelle Gemini et renvoie le texte (ou du JSON brut si json_mode).

    Réessaie une fois après une courte pause en cas de dépassement de quota (429),
    car le tier gratuit est limité à quelques requêtes par minute.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cle_gemini())
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    derniere = None
    for tentative in range(essais):
        try:
            reponse = client.models.generate_content(model=MODELE, contents=prompt, config=config)
            return reponse.text or ""
        except Exception as e:  # noqa: BLE001
            derniere = e
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and tentative < essais - 1:
                time.sleep(12)  # on attend que la fenêtre de quota (par minute) se libère
                continue
            raise
    raise derniere  # pragma: no cover


# ------------------------------------------------------------------
# Récupération (retrieval) : index TF-IDF mis en cache par dataset
# ------------------------------------------------------------------
def _index_recherche(df: pd.DataFrame):
    """Construit (une fois par dataset) le catalogue + l'index TF-IDF de recherche."""
    cle = f"_agent_index_{st.session_state.get('dataset_path')}"
    if cle not in st.session_state:
        catalogue = construire_catalogue_libelles(df)
        vect, matrice = construire_index_tfidf(catalogue, mode="Libelle seul")
        st.session_state[cle] = (catalogue, vect, matrice)
    return st.session_state[cle]


def exemples_similaires(df: pd.DataFrame, libelle: str, cible: str, k: int = 8,
                        col_categorie: str | None = None) -> list[dict]:
    """Retrouve les libellés proches déjà catégorisés (= contexte RAG).

    col_categorie : colonne d'où tirer la catégorie des exemples. Par défaut, la
    catégorie majoritaire d'origine (issue de l'index). Si on passe une colonne
    prédite (ex: 'Nature_predite'), les exemples reflètent la recatégorisation.
    """
    catalogue, vect, matrice = _index_recherche(df)
    voisins = rechercher_similaires(libelle, catalogue, vect, matrice, top_k=k)

    # Mapping libellé -> catégorie depuis la colonne demandée (cas "après recat").
    mapping = None
    if col_categorie and col_categorie in df.columns:
        labels = voisins["Libelle"].astype(str).str.strip().tolist()
        clef = df["Libelle"].astype(str).str.strip()
        sub = df[clef.isin(labels) & df[col_categorie].notna()].copy()
        if len(sub):
            sub["_l"] = sub["Libelle"].astype(str).str.strip()
            mapping = (sub.groupby("_l")[col_categorie]
                       .agg(lambda s: s.astype(str).mode().iloc[0]).to_dict())

    col_maj = f"{cible}_majoritaire"
    exemples = []
    for _, ligne in voisins.iterrows():
        lab = str(ligne["Libelle"]).strip()
        if mapping is not None:
            valeur = mapping.get(lab)
        else:
            valeur = ligne.get(col_maj) if col_maj in voisins.columns else None
        if pd.notna(valeur) and str(valeur).strip():
            exemples.append({
                "libelle": str(ligne["Libelle"]),
                "categorie": str(valeur).strip(),
                "score": round(float(ligne.get("score", 0) or 0), 3),
            })
    return exemples


# ------------------------------------------------------------------
# Cas d'usage : libellés sans catégorie (avant ou après recat)
# ------------------------------------------------------------------
def libelles_sans_categorie(df: pd.DataFrame, cible: str = "Nature",
                            source: str = "avant", top: int = 20) -> list[tuple[str, int]]:
    """Libellés (uniques, les plus fréquents) dont la `cible` est vide.

    source="avant" : la colonne d'origine est vide.
    source="apres" : la prédiction reste vide (l'algo n'a pas su trancher).
    Retourne [(libelle, nb_lignes), ...].
    """
    col = cible
    if source == "apres":
        cp = colonne_predite(cible)
        col = cp if cp in df.columns else cible
    if col not in df.columns or "Libelle" not in df.columns:
        return []
    manquantes = df[df[col].isna() & df["Libelle"].notna()]
    vc = manquantes["Libelle"].astype(str).str.strip().value_counts().head(top)
    return list(zip(vc.index.tolist(), vc.values.tolist()))


def _prompt_suggestion(libelle: str, cible: str, exemples: list[dict], existantes: list[str]) -> str:
    lignes_ex = "\n".join(f'- "{e["libelle"]}" → {e["categorie"]}' for e in exemples) or "(aucun exemple proche trouvé)"
    liste = ", ".join(existantes) if existantes else "(aucune)"
    return (
        f"Tu aides à catégoriser des produits e-commerce. On cherche la **{cible}** d'un produit.\n\n"
        f'Libellé à catégoriser : "{libelle}"\n\n'
        f"Produits similaires déjà catégorisés (récupérés automatiquement) :\n{lignes_ex}\n\n"
        f"{cible}s déjà utilisées dans le catalogue (extrait) : {liste}\n\n"
        "Règles :\n"
        f"- Choisis de préférence une {cible} EXISTANTE de la liste si elle convient au libellé.\n"
        f"- Si aucune ne convient vraiment, propose une NOUVELLE {cible} courte et cohérente.\n"
        "- Appuie-toi sur les exemples et le libellé ; n'invente pas de faits.\n\n"
        "Réponds STRICTEMENT en JSON avec ces clés :\n"
        '{"suggestion": "valeur proposée", "nouvelle": true|false, '
        '"confiance": "faible|moyenne|elevee", "justification": "phrase courte en français"}'
    )


def suggerer_categorie(df: pd.DataFrame, libelle: str, cible: str = "Nature", k: int = 8) -> dict:
    """Suggère une `cible` (existante ou nouvelle) pour un libellé, via RAG + Gemini."""
    exemples = exemples_similaires(df, libelle, cible, k=k)
    existantes = []
    if cible in df.columns:
        existantes = (df[cible].dropna().astype(str).str.strip()
                      .value_counts().head(80).index.tolist())
    prompt = _prompt_suggestion(libelle, cible, exemples, existantes)
    brut = _generer(prompt, json_mode=True)
    try:
        data = json.loads(brut)
    except Exception:
        data = {"suggestion": None, "nouvelle": None, "confiance": None,
                "justification": (brut or "").strip()[:300]}
    data["libelle"] = libelle
    data["exemples"] = exemples
    return data


def _prompt_lot(cible: str, blocs: list[str], existantes: list[str]) -> str:
    liste = ", ".join(existantes) if existantes else "(aucune)"
    corps = "\n".join(blocs)
    return (
        f"Tu catégorises des produits e-commerce. Pour CHAQUE libellé numéroté ci-dessous, "
        f"propose la **{cible}** la plus adaptée.\n\n"
        f"{corps}\n\n"
        f"{cible}s déjà utilisées dans le catalogue (extrait) : {liste}\n\n"
        "Règles :\n"
        f"- Privilégie une {cible} EXISTANTE de la liste si elle convient.\n"
        f"- Sinon propose une NOUVELLE {cible} courte et cohérente.\n"
        "- Appuie-toi sur les exemples proches fournis ; n'invente pas de faits.\n\n"
        "Réponds STRICTEMENT en JSON : un TABLEAU d'objets, UN par libellé, DANS LE MÊME ORDRE, "
        "chaque objet au format :\n"
        '{"suggestion": "...", "nouvelle": true|false, "confiance": "faible|moyenne|elevee", '
        '"justification": "phrase courte"}'
    )


def suggerer_categories_lot(df: pd.DataFrame, libelles: list[str], cible: str = "Nature",
                            source: str = "avant", k: int = 6) -> list[dict]:
    """Suggère une `cible` pour PLUSIEURS libellés en UN SEUL appel Gemini (économie de quota).

    source="apres" : les exemples et les catégories existantes proviennent de la colonne
    PRÉDITE (le LLM s'appuie alors sur la recatégorisation). Sinon, sur les valeurs d'origine.
    """
    apres = source == "apres" and colonne_predite(cible) in df.columns
    col_ref = colonne_predite(cible) if apres else cible
    col_cat = col_ref if apres else None  # None => catégorie majoritaire d'origine (index)

    blocs, exemples_par_lib = [], []
    for i, lib in enumerate(libelles, 1):
        ex = exemples_similaires(df, lib, cible, k=k, col_categorie=col_cat)
        exemples_par_lib.append(ex)
        ex_txt = " ; ".join(f'"{e["libelle"]}"→{e["categorie"]}' for e in ex) or "(aucun)"
        blocs.append(f'{i}. Libellé : "{lib}"\n   Exemples proches : {ex_txt}')

    existantes = []
    if col_ref in df.columns:
        existantes = (df[col_ref].dropna().astype(str).str.strip()
                      .value_counts().head(80).index.tolist())

    brut = _generer(_prompt_lot(cible, blocs, existantes), json_mode=True)
    try:
        data = json.loads(brut)
        if isinstance(data, dict):  # tolère {"resultats": [...]}
            data = data.get("resultats") or data.get("suggestions") or []
    except Exception:
        data = []

    resultats = []
    for i, lib in enumerate(libelles):
        item = data[i] if i < len(data) and isinstance(data[i], dict) else {}
        resultats.append({
            "libelle": lib,
            "suggestion": item.get("suggestion"),
            "nouvelle": item.get("nouvelle"),
            "confiance": item.get("confiance"),
            "justification": item.get("justification"),
            "exemples": exemples_par_lib[i],
        })
    return resultats


# ------------------------------------------------------------------
# Chat libre sur le dataset
# ------------------------------------------------------------------
def _resume_dataset(df: pd.DataFrame) -> str:
    parties = [f"{len(df):,} lignes"]
    if "Libelle" in df.columns:
        parties.append(f"{df['Libelle'].nunique():,} libellés uniques")
    for c in ("Nature", "Univers", "Vendeur"):
        if c in df.columns:
            parties.append(f"{df[c].nunique():,} {c.lower()}s, {int(df[c].isna().sum()):,} vides")
    return " · ".join(parties)


def repondre_chat(df: pd.DataFrame, question: str, k: int = 8) -> dict:
    """Répond à une question en s'appuyant sur un résumé du dataset + libellés récupérés."""
    catalogue, vect, matrice = _index_recherche(df)
    voisins = rechercher_similaires(question, catalogue, vect, matrice, top_k=k)
    cols = [c for c in ["Libelle", "Nature_majoritaire", "Univers_majoritaire", "nb_lignes"]
            if c in voisins.columns]
    contexte = voisins[cols].to_dict("records") if cols else []
    lignes_ctx = "\n".join(
        f"- {r.get('Libelle','')} | Nature: {r.get('Nature_majoritaire','?')} | "
        f"Univers: {r.get('Univers_majoritaire','?')}" for r in contexte
    ) or "(aucun libellé proche)"

    prompt = (
        "Tu es un assistant qui aide à analyser et fiabiliser un dataset e-commerce.\n"
        f"Résumé du dataset : {_resume_dataset(df)}.\n\n"
        f"Libellés du catalogue proches de la question (récupérés automatiquement) :\n{lignes_ctx}\n\n"
        f"Question de l'utilisateur : {question}\n\n"
        "Consignes :\n"
        "- Réponds en français, de façon concise et factuelle, à partir du contexte fourni.\n"
        "- Pour une suggestion de catégorie, propose une Nature/Univers existante ou, si rien ne convient, une nouvelle.\n"
        "- Pour un calcul précis (totaux, classements, pourcentages), indique d'utiliser les pages d'analyse dédiées : "
        "tu n'as pas accès à l'ensemble des lignes, seulement à un échantillon récupéré.\n"
    )
    return {"reponse": _generer(prompt), "contexte": contexte}
