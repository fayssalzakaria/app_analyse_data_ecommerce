"""Page 8 - Assistant IA (RAG + Gemini).

Deux usages :
1. Suggérer une Nature/Univers pour les libellés SANS catégorie (avant ou après recat),
   en s'appuyant sur les libellés similaires déjà catégorisés (RAG).
2. Poser des questions libres sur le dataset.

Nécessite une clé GEMINI_API_KEY (dans .streamlit/secrets.toml en local, ou en secret
sur Hugging Face).
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import enrich, agent
from core.recat import colonne_predite

st.set_page_config(page_title="Assistant IA", page_icon="🤖", layout="wide")
st.title("🤖 Assistant IA")
st.caption("Aide à la catégorisation et questions sur le dataset (RAG + Gemini).")

# --- Vérifications préalables -------------------------------------------------
if not agent.gemini_pret():
    st.error(
        "Clé Gemini introuvable. Ajoute `GEMINI_API_KEY` dans `.streamlit/secrets.toml` "
        "(en local) ou dans les secrets du Space Hugging Face, puis recharge."
    )
    st.stop()

df = enrich.obtenir_df_travail()
if df is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()
if "Libelle" not in df.columns:
    st.warning("La colonne `Libelle` est nécessaire à l'assistant.")
    st.stop()

st.info(f"Modèle : `{agent.MODELE}` · Dataset : {len(df):,} lignes. "
        "L'assistant s'appuie sur les libellés similaires déjà catégorisés (il ne voit pas tout le dataset).")

onglet_sugg, onglet_chat = st.tabs(["🏷️ Suggérer des catégories", "💬 Chat"])

# ============================================================
# 1) Suggestions pour les libellés sans catégorie
# ============================================================
with onglet_sugg:
    st.subheader("Suggérer une catégorie pour les libellés sans Nature/Univers")
    cible = st.radio("Catégorie à suggérer", ["Nature", "Univers"], horizontal=True)

    # « Après recat » n'est proposé que si la recatégorisation de cette cible a été lancée.
    recat_faite = colonne_predite(cible) in df.columns
    c2, c3 = st.columns(2)
    options_quand = ["avant"] + (["apres"] if recat_faite else [])
    source = c2.radio(
        "Quand ?", options_quand,
        format_func=lambda s: "Avant recat (valeur d'origine vide)" if s == "avant"
        else "Après recat (prédiction vide)",
    )
    if not recat_faite:
        c2.caption(f"« Après recat » s'activera une fois la **{cible}** recatégorisée (page 🎯).")
    nb = c3.slider("Nombre de libellés à traiter", 1, 20, 8,
                   help="Tout le lot part en 1 seul appel Gemini.")

    manquants = agent.libelles_sans_categorie(df, cible=cible, source=source, top=nb)
    if not manquants:
        st.success(f"Aucun libellé sans {cible} pour ce filtre. 🎉")
    else:
        st.caption(f"{len(manquants)} libellé(s) sans {cible} (les plus fréquents). "
                   "Tout le lot est traité en **un seul appel** (quota gratuit : 5 requêtes/min).")
        if st.button(f"💡 Suggérer une {cible} pour ces libellés", type="primary"):
            libelles = [lib for lib, _ in manquants]
            nb_par_lib = dict(manquants)
            with st.spinner("Suggestions en cours (1 appel Gemini)…"):
                try:
                    res = agent.suggerer_categories_lot(df, libelles, cible=cible, source=source)
                except Exception as e:  # quota, réseau...
                    res = None
                    st.error(f"Appel Gemini échoué : {e}")
            if res:
                lignes = [{
                    "Libellé": r["libelle"],
                    "nb lignes": nb_par_lib.get(r["libelle"], 0),
                    f"{cible} suggérée": r.get("suggestion"),
                    "nouvelle ?": "🆕 oui" if r.get("nouvelle") else "existante",
                    "confiance": r.get("confiance"),
                    "justification": r.get("justification"),
                } for r in res]
                st.session_state["agent_suggestions"] = pd.DataFrame(lignes)

        if "agent_suggestions" in st.session_state:
            st.dataframe(st.session_state["agent_suggestions"],
                         use_container_width=True, hide_index=True)
            st.caption("⚠️ Suggestions = aide à la décision. Valide avant de les appliquer au dataset.")

# ============================================================
# 2) Chat libre
# ============================================================
with onglet_chat:
    st.subheader("Poser une question sur le dataset")
    st.caption("Ex : « propose une nature pour 'tapis salon 200x300' », "
               "« quels libellés ressemblent à un canapé d'angle ? ».")

    historique = st.session_state.setdefault("agent_chat", [])
    for msg in historique:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ta question…")
    if question:
        historique.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Réflexion…"):
                try:
                    res = agent.repondre_chat(df, question)
                    reponse = res["reponse"]
                    contexte = res.get("contexte", [])
                except Exception as e:
                    reponse, contexte = f"Erreur lors de l'appel Gemini : {e}", []
            st.markdown(reponse)
            if contexte:
                with st.expander("Sources récupérées (libellés du catalogue)"):
                    st.dataframe(pd.DataFrame(contexte), use_container_width=True, hide_index=True)
        historique.append({"role": "assistant", "content": reponse})

    if historique and st.button("🗑️ Effacer la conversation"):
        st.session_state["agent_chat"] = []
        st.rerun()
