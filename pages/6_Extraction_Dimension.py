"""Page 6 - Extraction des dimensions depuis les libellés (calcul à la demande)."""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import appliquer_filtres_globaux, afficher_filtres_sidebar
from core import enrich


st.set_page_config(page_title="Extraction dimension", page_icon="📐", layout="wide")
st.title("📐 Extraction des dimensions")

df_full = enrich.obtenir_df_travail()
if df_full is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

with st.expander("⚙️ Extraction des dimensions",
                 expanded=not enrich.possede_colonnes(["Dimension_Commentaire"])):
    if st.button("📐 Extraire les dimensions", type="primary"):
        with st.spinner("Extraction des dimensions sur les libellés..."):
            info = enrich.calculer_dimension()
        if info.get("ok"):
            st.success(
                f"Dimensions extraites : {info['n_avec_dim']:,} lignes avec dimension "
                f"({info['n_avec_dim'] / info['n_total'] * 100:.1f}%), {info['n_diam']:,} diamètres."
            )
            st.rerun()
        else:
            st.error(info.get("msg", "Échec de l'extraction."))

if not enrich.possede_colonnes(["Dimension_Commentaire", "dim_label"]):
    st.info("👆 Lance l'extraction des dimensions pour calculer les colonnes dimension sur le fichier chargé.")
    st.stop()

df_full = enrich.obtenir_df_travail()
afficher_filtres_sidebar(df_full)
df = appliquer_filtres_globaux(df_full)

st.markdown(
    """
L'extraction utilise l'algo **V2** (`extract_dimension_v2`) avec :
- réparation des décimales-espace (`50 0 x 50 0` → `50.0 x 50.0`)
- regex 2D / 3D avec conversion d'unités (mm, cm, m, inch)
- **diamètre** (Ø, diam.)
- **cote isolée** (`dimension_simple_cm`, ex: une seule mesure « 140 cm »)
- rejet faux positifs (résolutions écran, puissances, références).
"""
)

# KPIs
n_total = len(df)
n_dim = int(df["Dimension_Commentaire"].isin(
    ["1 dimension", "2 dimensions", "3 dimensions", "Diamètre", "Avec alerte"]
).sum())
n_diam = int(df["diametre_cm"].notna().sum()) if "diametre_cm" in df.columns else 0
n_simple = int(df["dimension_simple_cm"].notna().sum()) if "dimension_simple_cm" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes total", f"{n_total:,}")
c2.metric("Lignes avec dimension", f"{n_dim:,}", delta=f"{n_dim/n_total*100:.2f}%")
c3.metric("Diamètres détectés", f"{n_diam:,}")
c4.metric("Cotes isolées", f"{n_simple:,}")

# Distribution Dimension_Commentaire
st.subheader("Distribution Dimension_Commentaire")
order_dim = ["Aucune dimension", "1 dimension", "2 dimensions", "3 dimensions",
              "Diamètre", "Avec alerte"]
colors_dim = {"Aucune dimension": "#cccccc", "1 dimension": "#4daf4a",
               "2 dimensions": "#377eb8", "3 dimensions": "#984ea3",
               "Diamètre": "#ff7f00", "Avec alerte": "#d62728"}
counts = df["Dimension_Commentaire"].value_counts().reindex(order_dim).fillna(0).astype(int)
counts = counts[counts > 0]
fig = px.bar(
    x=counts.index, y=counts.values,
    labels={"x": "Catégorie", "y": "Nombre de lignes"},
    text=[f"{v:,}<br>({v/n_total*100:.2f}%)" for v in counts.values],
    color=counts.index, color_discrete_map=colors_dim,
)
fig.update_traces(textposition="outside", showlegend=False)
fig.update_layout(height=420, xaxis_tickangle=-15)
st.plotly_chart(fig, use_container_width=True)

# Top 25 dim_label
st.subheader("Top 25 dimensions (`dim_label`)")
top_dims = df["dim_label"].dropna().value_counts().head(25)
fig = px.bar(
    x=top_dims.values, y=top_dims.index, orientation="h",
    labels={"x": "Nombre de lignes", "y": "dim_label"},
    text=[f"{v:,}" for v in top_dims.values],
)
fig.update_traces(textposition="outside", marker_color="#5b8db8")
fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# Dimension simple par type de contexte
if "dimension_simple_type" in df.columns:
    st.subheader("Cotes isolées par contexte")
    types = df["dimension_simple_type"].dropna().value_counts().head(15)
    fig = px.bar(
        x=types.index, y=types.values,
        labels={"x": "Contexte détecté", "y": "Nombre de lignes"},
        text=[f"{v:,}" for v in types.values],
    )
    fig.update_traces(textposition="outside", marker_color="#9ec39e")
    fig.update_layout(height=350, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

# Repartition par Vendeur
st.subheader("Dimension_Commentaire par Vendeur")
if "Vendeur" in df.columns:
    piv = pd.crosstab(df["Vendeur"], df["Dimension_Commentaire"], normalize="index") * 100
    piv = piv.round(2)
    order_cols = [c for c in order_dim if c in piv.columns]
    piv = piv[order_cols]
    st.dataframe(piv, use_container_width=True)
else:
    st.info("Colonne `Vendeur` absente : répartition par vendeur indisponible.")

# Audits
with st.expander("Audit : lignes 'Avec alerte'", expanded=False):
    alertes = df[df["Dimension_Commentaire"] == "Avec alerte"]
    st.write(f"**{len(alertes):,}** lignes avec alerte (réparation décimale ou valeur suspecte).")
    if len(alertes) > 0:
        cols_show = [c for c in ["Libelle", "dim_label", "L_cm", "l_cm", "H_cm",
                                   "dimension_warning", "Vendeur"] if c in alertes.columns]
        sample = alertes[cols_show].drop_duplicates(subset=["Libelle"]).head(30)
        st.dataframe(sample, use_container_width=True, hide_index=True)

with st.expander("Audit : diamètres détectés", expanded=False):
    diams = df[df["diametre_cm"].notna()] if "diametre_cm" in df.columns else pd.DataFrame()
    st.write(f"**{len(diams):,}** lignes avec diamètre détecté.")
    if len(diams) > 0:
        cols_show = [c for c in ["Libelle", "diametre_cm", "H_cm", "dim_label",
                                   "Vendeur"] if c in diams.columns]
        sample = diams[cols_show].drop_duplicates(subset=["Libelle"]).head(30)
        st.dataframe(sample, use_container_width=True, hide_index=True)
