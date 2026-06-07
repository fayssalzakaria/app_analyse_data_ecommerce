"""Application Streamlit - Analyse et purification d'un dataset (catégorisation Nature, extraction d'attributs)."""
import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import (
    charger_dataset,
    afficher_filtres_sidebar,
    reinitialiser_si_changement_dataset,
    recharger_dataset_courant,
    reinitialiser_tout,
)
from core.metrics import statistiques_globales


st.set_page_config(
    page_title="Analyse & purification de dataset de e_commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("📊 Analyse & purification de dataset")
    st.caption("Exploration, correction des catégories et extraction d'attributs à partir d'un fichier (Libellé, Nature, etc.).")

    with st.sidebar:
        st.markdown("### Fichier source")
        # Aucun fichier n'est chargé par défaut : l'utilisateur fournit le sien.
        # Clé dynamique : l'incrémenter (au reset) vide réellement le widget d'upload.
        st.session_state.setdefault("uploader_key", 0)
        uploaded = st.file_uploader(
            "Charger un fichier (.xlsb, .xlsx, .csv, .parquet)",
            type=["xlsb", "xlsx", "csv", "parquet"],
            key=f"file_uploader_{st.session_state['uploader_key']}",
        )
        if uploaded is not None:
            # On ne recrée le fichier temporaire que si l'upload a changé
            # (sinon dataset_path changerait à chaque rerun et purgerait les calculs).
            file_sig = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("_uploaded_sig") != file_sig:
                suffix = os.path.splitext(uploaded.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.getbuffer())
                tmp.close()
                st.session_state["dataset_path"] = tmp.name
                st.session_state["_uploaded_sig"] = file_sig

        # Jeu de démonstration synthétique : chargement explicite (jamais automatique).
        _sample = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "data", "sample", "sample_ecommerce.csv")
        )
        if os.path.exists(_sample):
            if st.button("Charger le jeu de démo"):
                st.session_state["dataset_path"] = _sample
                st.session_state.pop("_uploaded_sig", None)
            st.caption("Jeu synthétique fourni, pour tester l'app sans données réelles.")

        # Si le fichier source a changé, on purge tout l'état dérivé (enrichissements,
        # filtres, export, caches de pages).
        if reinitialiser_si_changement_dataset():
            st.toast("Nouveau dataset chargé : état précédent réinitialisé.")

        b1, b2 = st.columns(2)
        if b1.button("🔄 Recharger", help="Recalcule à partir du fichier courant (vide son cache)."):
            recharger_dataset_courant()
            st.rerun()
        if b2.button("🗑️ Tout réinitialiser",
                     help="Décharge le fichier ET supprime tout le cache (disque + mémoire)."):
            _next_key = st.session_state.get("uploader_key", 0) + 1
            reinitialiser_tout()
            st.session_state["uploader_key"] = _next_key  # nouveau widget d'upload vide
            st.rerun()

    path = st.session_state.get("dataset_path")
    if not path:
        st.info("👈 Sélectionne un fichier dans la sidebar pour démarrer.")
        st.stop()

    st.success("Application demarree. Les caches disque existants sont conserves.")
    st.caption(
        "Pour eviter un chargement lourd au demarrage, l'accueil ne charge pas le dataset "
        "tant que tu ne le demandes pas explicitement. Utilise le menu de gauche pour aller "
        "directement sur une page d'analyse."
    )

    st.markdown(
        """
### Pages disponibles

| Page | Usage |
|---|---|
| **0. Recherche sémantique** | Recherche de libellés proches (TF-IDF) + aide à la saisie |
| **1. Données brutes** | Stats globales : KPI, évolution, répartitions, qualité |
| **2. Visualisation graphe** | Graphe relationnel + nuage 2D/3D + Sankey + avant/après |
| **3. Algo catégorisation Nature** | Recatégorisation calculée dans l'app (1 ou 2 passes) |
| **4. Effet correction** | Cramér's V avant/après + matrices par vendeur |
| **5. Extraction couleur** | Couleur dominante extraite du libellé |
| **6. Extraction dimension** | Format LxlxH, diamètre, cote isolée en cm |
| **7. Export** | Téléchargement du dataset enrichi (CSV / Excel / Parquet) |

L'enrichissement (catégorisation, couleur, dimension) se calcule à la demande
depuis les pages dédiées, sur le fichier chargé.
        """
    )

    if not st.button("Charger l'aperçu dataset sur l'accueil"):
        st.stop()

    with st.spinner("Chargement du dataset..."):
        df = charger_dataset(path)

    afficher_filtres_sidebar(df)

    st.success(f"Dataset chargé : **{len(df):,}** lignes, **{len(df.columns)}** colonnes.")

    stats = statistiques_globales(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{stats['nb_lignes']:,}")
    c2.metric("Commandes uniques", f"{stats.get('nb_commandes_uniques') or 0:,}")
    c3.metric("CA total", f"{stats.get('ca_total', 0):,.0f} €")
    c4.metric("Période", (
        f"{stats['periode'][0].date()} → {stats['periode'][1].date()}"
        if stats.get("periode") else "n/a"
    ))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Vendeurs", stats["nb_vendeurs"] or 0)
    c6.metric("Univers", stats["nb_univers"] or 0)
    c7.metric("Natures", stats["nb_natures"] or 0)
    c8.metric("Lignes Nature vide", f"{stats['nan_nature']:,}")

    st.divider()

    st.markdown("### Aperçu des données")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    with st.expander("Schéma des colonnes"):
        info = []
        for c in df.columns:
            info.append({
                "Colonne": c,
                "Type": str(df[c].dtype),
                "Non-nuls": int(df[c].notna().sum()),
                "Nuls": int(df[c].isna().sum()),
                "Uniques": int(df[c].nunique()),
            })
        st.dataframe(info, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
