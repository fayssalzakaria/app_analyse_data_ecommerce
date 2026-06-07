"""Page 5 - Extraction des couleurs depuis les libellés (calcul à la demande).

Les couleurs sont extraites par le bouton de la page (voir core/extract.py), puis
on visualise leur répartition (distribution, top couleurs, vue par vendeur).
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import appliquer_filtres_globaux, afficher_filtres_sidebar
from core import enrich


st.set_page_config(page_title="Extraction couleur", page_icon="🎨", layout="wide")
st.title("🎨 Extraction des couleurs")

df_full = enrich.obtenir_df_travail()
if df_full is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

with st.expander("⚙️ Extraction des couleurs",
                 expanded=not enrich.possede_colonnes(["couleur_extraite"])):
    fine = st.checkbox("Palette fine (nuances composées : bleu marine, chêne clair...)", value=True)
    if st.button("🎨 Extraire les couleurs", type="primary"):
        with st.spinner("Extraction des couleurs sur les libellés..."):
            info = enrich.calculer_couleur(fine=fine)
        if info.get("ok"):
            st.success(
                f"Couleurs extraites : {info['n_avec_couleur']:,} lignes avec couleur "
                f"({info['n_avec_couleur'] / info['n_total'] * 100:.1f}%), "
                f"{info['n_multi']:,} multi-couleur."
            )
            st.rerun()
        else:
            st.error(info.get("msg", "Échec de l'extraction."))

if not enrich.possede_colonnes(["couleur_extraite", "Couleur_Commentaire", "nb_couleurs_detectees"]):
    st.info("👆 Lance l'extraction des couleurs pour calculer les colonnes couleur sur le fichier chargé.")
    st.stop()

df_full = enrich.obtenir_df_travail()
afficher_filtres_sidebar(df_full)
df = appliquer_filtres_globaux(df_full)

st.markdown(
    """
L'extraction utilise l'algo à **richesse maximale** (`extract_color_contextual`) :
- palette de 27 couleurs de base + 27 nuances composées (`bleu marine`, `chêne clair`...)
- 8 couleurs additionnelles (`terracotta`, `camel`, `sable`, `lin`, `noyer`, `teck`, `acacia`, `bois clair`)
- aliases de correction typo / troncatures (`blan→blanc`, `noi→noir`, `girs/grsi→gris`, `ch ne→chene`, `antracite→anthracite`)
- multi-couleur (compte toutes les couleurs distinctes par libellé)
- détection contextuelle niveau 3 (chêne couleur vs chêne matière).
"""
)

# ============================================================
# KPIs
# ============================================================
n_total = len(df)
n_color = int(df["couleur_extraite"].notna().sum())
n_multi = int((df["nb_couleurs_detectees"] >= 2).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes total", f"{n_total:,}")
c2.metric("Lignes avec couleur", f"{n_color:,}", delta=f"{n_color/n_total*100:.2f}%")
c3.metric("Lignes multi-couleur", f"{n_multi:,}", delta=f"{n_multi/n_total*100:.2f}%")
c4.metric("Couleurs distinctes détectées", df["couleur_extraite"].nunique(dropna=True))

# ============================================================
# Distribution Couleur_Commentaire
# ============================================================
st.subheader("Distribution Couleur_Commentaire")
order_couleur = ["Aucune couleur", "Couleur unique", "2 couleurs", "3 couleurs",
                  "4 couleurs", "5 couleurs", "6 couleurs", "7 couleurs", "8 couleurs"]
counts = df["Couleur_Commentaire"].value_counts().reindex(order_couleur).fillna(0).astype(int)
counts = counts[counts > 0]
fig = px.bar(
    x=counts.index, y=counts.values,
    labels={"x": "Catégorie", "y": "Nombre de lignes"},
    text=[f"{v:,}<br>({v/n_total*100:.2f}%)" for v in counts.values],
    color=counts.index,
    color_discrete_sequence=px.colors.qualitative.Set3,
)
fig.update_traces(textposition="outside", showlegend=False)
fig.update_layout(height=420, xaxis_tickangle=-20)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Top 25 couleurs
# ============================================================
st.subheader("Top 25 couleurs détectées (`couleur_extraite`)")
top_colors = df["couleur_extraite"].dropna().value_counts().head(25)
fig = px.bar(
    x=top_colors.values, y=top_colors.index,
    orientation="h",
    labels={"x": "Nombre de lignes", "y": "Couleur"},
    text=[f"{v:,}" for v in top_colors.values],
)
fig.update_traces(textposition="outside", marker_color="#7b9e87")
fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Exemples multi-couleur
# ============================================================
st.subheader("Exemples multi-couleur")
nb_choisi = st.slider("Nombre de couleurs à voir", min_value=2, max_value=8, value=2)
sub = df[df["nb_couleurs_detectees"] == nb_choisi].copy()
if len(sub) > 0:
    libs_uniques = sub["Libelle"].dropna().drop_duplicates().head(20)
    examples = []
    for lib in libs_uniques:
        row = sub[sub["Libelle"] == lib].iloc[0]
        examples.append({
            "Libelle": lib,
            "Couleurs détectées": row.get("couleurs_toutes", ""),
            "Vendeur": row.get("Vendeur", ""),
            "nb": row.get("nb_couleurs_detectees", 0),
        })
    st.dataframe(pd.DataFrame(examples), use_container_width=True, hide_index=True)
else:
    st.info(f"Aucun libellé avec exactement {nb_choisi} couleurs détectées dans le filtre actuel.")

# ============================================================
# Repartition par Vendeur
# ============================================================
st.subheader("Couleur_Commentaire par Vendeur")
if "Vendeur" in df.columns:
    piv = pd.crosstab(df["Vendeur"], df["Couleur_Commentaire"], normalize="index") * 100
    piv = piv.round(2)
    order_cols = [c for c in order_couleur if c in piv.columns]
    piv = piv[order_cols]
    st.dataframe(piv, use_container_width=True)
else:
    st.info("Colonne `Vendeur` absente : répartition par vendeur indisponible.")

# ============================================================
# Audit cases ambigus
# ============================================================
with st.expander("Cas ambigus / chêne (matière vs couleur)", expanded=False):
    if "couleur_decision" in df.columns:
        ambigus = df[df["couleur_decision"].astype(str).str.contains("ambigu", case=False, na=False)]
        st.write(f"**{len(ambigus):,}** libellés en statut ambigu (chêne, noyer, teck...).")
        if len(ambigus) > 0:
            sample = ambigus[["Libelle", "couleur_extraite", "couleur_decision", "Vendeur"]].drop_duplicates(subset=["Libelle"]).head(30)
            st.dataframe(sample, use_container_width=True, hide_index=True)
    else:
        st.info("Colonne `couleur_decision` absente du livrable.")
