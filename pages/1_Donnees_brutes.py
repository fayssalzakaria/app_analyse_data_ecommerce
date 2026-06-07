"""Page 1 - Stats globales."""
import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import obtenir_dataframe_actif, afficher_filtres_sidebar, appliquer_filtres_globaux
from core.metrics import qualite_par_vendeur

st.set_page_config(page_title="Stats globales", page_icon="📊", layout="wide")
st.title("📊 Stats globales")

df = obtenir_dataframe_actif()
if df is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

afficher_filtres_sidebar(df)
df = appliquer_filtres_globaux(df)

# KPI haut de page
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df):,}")
c2.metric("Commandes uniques", f"{df['Cod_cmd'].nunique():,}" if "Cod_cmd" in df.columns else "n/a")
c3.metric("CA total", f"{df['CA'].sum():,.0f} €" if "CA" in df.columns else "n/a")
c4.metric("Qté totale", f"{df['Quantite'].sum():,.0f}" if "Quantite" in df.columns else "n/a")

if "Date" in df.columns and df["Date"].notna().any():
    c5, c6 = st.columns(2)
    c5.metric("Premier achat", df["Date"].min().date().isoformat())
    c6.metric("Dernier achat", df["Date"].max().date().isoformat())

st.divider()

# ---- Évolution temporelle ----
st.subheader("Évolution temporelle")
if "Date" in df.columns and df["Date"].notna().any():
    df_t = df.dropna(subset=["Date"]).copy()
    df_t["mois"] = df_t["Date"].dt.to_period("M").dt.to_timestamp()
    agg = df_t.groupby("mois").agg(
        nb_lignes=("Libelle", "size"),
        nb_commandes=("Cod_cmd", "nunique") if "Cod_cmd" in df_t.columns else ("Libelle", "size"),
        qte=("Quantite", "sum") if "Quantite" in df_t.columns else ("Libelle", "size"),
        ca=("CA", "sum") if "CA" in df_t.columns else ("Libelle", "size"),
    ).reset_index()

    metric = st.selectbox("Variable à tracer", ["nb_lignes", "nb_commandes", "qte", "ca"], index=3)
    fig = px.line(agg, x="mois", y=metric, markers=True, title=f"Évolution mensuelle - {metric}")
    fig.update_layout(height=380, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pas de colonne 'Date' exploitable.")

# ---- Répartition par vendeur ----
st.subheader("Répartition par vendeur")
if "Vendeur" in df.columns:
    by_v = df.groupby("Vendeur").agg(
        nb_lignes=("Libelle", "size"),
        ca=("CA", "sum") if "CA" in df.columns else ("Libelle", "size"),
        nb_cmd=("Cod_cmd", "nunique") if "Cod_cmd" in df.columns else ("Libelle", "size"),
    ).reset_index().sort_values("nb_lignes", ascending=False)
    by_v["panier_moyen"] = (by_v["ca"] / by_v["nb_cmd"]).round(0)

    colA, colB = st.columns(2)
    fig_v = px.bar(by_v, x="Vendeur", y="nb_lignes", title="Nombre de lignes par vendeur", text_auto=True)
    colA.plotly_chart(fig_v, use_container_width=True)
    fig_ca = px.bar(by_v, x="Vendeur", y="ca", title="CA par vendeur", text_auto=".2s")
    colB.plotly_chart(fig_ca, use_container_width=True)

    st.dataframe(by_v, use_container_width=True, hide_index=True)

# ---- Qualité de donnée par vendeur ----
st.subheader("Qualité de donnée par vendeur")
q = qualite_par_vendeur(df)
if not q.empty:
    st.dataframe(q, use_container_width=True)
    fig_q = px.bar(
        q.reset_index(),
        x="Vendeur",
        y=["pct_nan_univers", "pct_nan_nature"],
        barmode="group",
        title="% de valeurs manquantes par vendeur",
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ---- Répartition catégories actuelle ----
st.subheader("Répartition des catégories actuelles (AVANT correction)")

tabs = st.tabs(["Univers", "Top 30 Natures", "Heatmap Univers × Vendeur"])

with tabs[0]:
    if "Univers" in df.columns:
        u = df["Univers"].fillna("(vide)").value_counts().reset_index()
        u.columns = ["Univers", "nb_lignes"]
        fig = px.bar(u, x="nb_lignes", y="Univers", orientation="h", title="Répartition par Univers", text_auto=True)
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    if "Nature" in df.columns:
        n = df["Nature"].fillna("(vide)").value_counts().head(30).reset_index()
        n.columns = ["Nature", "nb_lignes"]
        fig = px.bar(n, x="nb_lignes", y="Nature", orientation="h", title="Top 30 Natures", text_auto=True)
        fig.update_layout(height=700, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    if {"Univers", "Vendeur"}.issubset(df.columns):
        ct = pd.crosstab(df["Univers"].fillna("(vide)"), df["Vendeur"])
        fig = px.imshow(ct, text_auto=True, aspect="auto", title="Nb lignes Univers × Vendeur")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---- Top produits ----
st.subheader("Top produits")
colT1, colT2 = st.columns(2)
if "CA" in df.columns:
    top_ca = df.groupby("Libelle")["CA"].sum().nlargest(20).reset_index()
    fig = px.bar(top_ca, x="CA", y="Libelle", orientation="h", title="Top 20 par CA")
    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    colT1.plotly_chart(fig, use_container_width=True)
top_vol = df["Libelle"].value_counts().head(20).reset_index()
top_vol.columns = ["Libelle", "nb_lignes"]
fig = px.bar(top_vol, x="nb_lignes", y="Libelle", orientation="h", title="Top 20 par volume")
fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
colT2.plotly_chart(fig, use_container_width=True)

# ---- Saisonnalité ----
st.subheader("Saisonnalité - heatmap mois × vendeur")
if {"Date", "Vendeur"}.issubset(df.columns) and df["Date"].notna().any():
    df_t = df.dropna(subset=["Date"]).copy()
    df_t["mois"] = df_t["Date"].dt.to_period("M").astype(str)
    pivot = df_t.pivot_table(index="Vendeur", columns="mois", values="Libelle", aggfunc="size", fill_value=0)
    fig = px.imshow(pivot, aspect="auto", title="Nb lignes / vendeur / mois")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
