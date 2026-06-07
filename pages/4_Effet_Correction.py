"""Page 4 - Effet de la correction et stats par Vendeur.

Compare le dataset AVANT (Nature originale) et APRES (Nature_predite) :
- Cramer V Vendeur ↔ Nature
- Matrice vendeur × Natures suspectes
- Taux de correction par Vendeur (avec code couleur rouge/orange/vert)
- Repartition couleur par Vendeur (stacked 100%)
- Repartition dimension par Vendeur (stacked 100%)
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import appliquer_filtres_globaux, afficher_filtres_sidebar
from core import enrich


st.set_page_config(page_title="Effet correction", page_icon="📈", layout="wide")
st.title("📈 Effet de la correction et stats par Vendeur")

df_full = enrich.obtenir_df_travail()
if df_full is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

if not enrich.possede_colonnes(["Nature_predite"]):
    st.info(
        "Cette page compare la Nature **avant / après** recatégorisation. "
        "Lance d'abord la **recatégorisation** sur la page « 🎯 Algo catégorisation Nature »."
    )
    st.stop()

afficher_filtres_sidebar(df_full)
df = appliquer_filtres_globaux(df_full)

# ============================================================
# 1. Matrice vendeur × Natures suspectes : avant vs apres
# ============================================================
st.subheader("1. Matrice Vendeur × Natures analysées")

# Ordre des vendeurs : par volume décroissant (dynamique, indépendant du dataset).
ordre_vendeurs = (
    df["Vendeur"].dropna().astype(str).str.strip().value_counts().index.tolist()
    if "Vendeur" in df.columns else []
)


def _changed_natures(df_local):
    """Natures les plus souvent corrigées par l'algo (Nature -> Nature_predite)."""
    s = df_local.dropna(subset=["Nature", "Nature_predite"]).copy()
    s = s[s["Nature"].astype(str).str.strip() != s["Nature_predite"].astype(str).str.strip()]
    if s.empty:
        return []
    return (
        pd.concat([s["Nature"], s["Nature_predite"]]).astype(str).str.strip()
        .value_counts().index.tolist()
    )


_all_natures = sorted(
    set(df["Nature"].dropna().astype(str).str.strip())
    | set(df["Nature_predite"].dropna().astype(str).str.strip())
)
_default_natures = (
    _changed_natures(df)
    or df["Nature"].dropna().astype(str).str.strip().value_counts().index.tolist()
)[:6]
SUSPECT_NATURES = st.multiselect(
    "Natures à analyser dans la matrice",
    _all_natures,
    default=[n for n in _default_natures if n in _all_natures],
    help="Par défaut : les Natures les plus souvent corrigées par l'algo. Tu peux ajuster la sélection.",
)
if not SUSPECT_NATURES:
    st.info("Sélectionne au moins une Nature pour afficher la matrice.")


def matrice_pourcentage(df_local, col_nature):
    s = df_local.dropna(subset=["Vendeur", col_nature]).copy()
    s["Vendeur"] = s["Vendeur"].astype(str).str.strip()
    s_sus = s[s[col_nature].isin(SUSPECT_NATURES)]
    ct = pd.crosstab(s_sus["Vendeur"], s_sus[col_nature])
    totals = s.groupby("Vendeur").size()
    pct = ct.div(totals.reindex(ct.index).values, axis=0) * 100
    pct = pct.reindex(columns=SUSPECT_NATURES, fill_value=0)
    pct = pct.reindex([v for v in ordre_vendeurs if v in pct.index]).fillna(0)
    return pct.round(2)

c1, c2 = st.columns(2)
with c1:
    st.caption("**Avant correction** (Nature originale)")
    pct_av = matrice_pourcentage(df, "Nature")
    fig = px.imshow(pct_av.values, x=pct_av.columns, y=pct_av.index,
                     color_continuous_scale="Reds", aspect="auto",
                     text_auto=".1f", labels=dict(color="%"))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.caption("**Après correction** (Nature_predite)")
    pct_ap = matrice_pourcentage(df, "Nature_predite")
    fig = px.imshow(pct_ap.values, x=pct_ap.columns, y=pct_ap.index,
                     color_continuous_scale="Reds", aspect="auto",
                     text_auto=".1f", labels=dict(color="%"))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2. Cramer V Vendeur ↔ Categorie : avant vs apres
# ============================================================
st.subheader("2. Force d'association Vendeur ↔ Catégorie (Cramer's V)")

def v_de_cramer(s1, s2):
    ct = pd.crosstab(s1, s2)
    if ct.size == 0 or ct.values.sum() == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, c = ct.shape
    denom = n * (min(r, c) - 1) if min(r, c) > 1 else 1
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0

sub = df.dropna(subset=["Vendeur"]).copy()
sub["Vendeur"] = sub["Vendeur"].astype(str).str.strip()

rows = []
for label, col_av, col_ap in [
    ("Univers", "Univers", "Univers"),
    ("Nature", "Nature", "Nature_predite"),
]:
    v_av = v_de_cramer(sub.dropna(subset=[col_av])["Vendeur"], sub.dropna(subset=[col_av])[col_av])
    v_ap = v_de_cramer(sub.dropna(subset=[col_ap])["Vendeur"], sub.dropna(subset=[col_ap])[col_ap])
    rows.append({"Variable": label, "Avant": v_av, "Apres": v_ap, "Delta": v_ap - v_av})

couples_av = sub.dropna(subset=["Univers", "Nature"]).copy()
couples_av["c"] = couples_av["Univers"].astype(str) + " / " + couples_av["Nature"].astype(str)
couples_ap = sub.dropna(subset=["Univers", "Nature_predite"]).copy()
couples_ap["c"] = couples_ap["Univers"].astype(str) + " / " + couples_ap["Nature_predite"].astype(str)
v_av = v_de_cramer(couples_av["Vendeur"], couples_av["c"])
v_ap = v_de_cramer(couples_ap["Vendeur"], couples_ap["c"])
rows.append({"Variable": "Couple Univers/Nature", "Avant": v_av, "Apres": v_ap, "Delta": v_ap - v_av})

cramer_df = pd.DataFrame(rows)

# Barchart compare
fig = go.Figure()
fig.add_trace(go.Bar(name="Avant", x=cramer_df["Variable"], y=cramer_df["Avant"], marker_color="#c44e52",
                     text=[f"{v:.3f}" for v in cramer_df["Avant"]], textposition="outside"))
fig.add_trace(go.Bar(name="Après", x=cramer_df["Variable"], y=cramer_df["Apres"], marker_color="#55a868",
                     text=[f"{v:.3f}" for v in cramer_df["Apres"]], textposition="outside"))
fig.update_layout(barmode="group", height=450, yaxis_title="Cramer's V")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(cramer_df.round(4), use_container_width=True, hide_index=True)

# ============================================================
# 3. Taux de correction par Vendeur
# ============================================================
st.subheader("3. Taux de correction Nature par Vendeur")
st.caption("🔴 rouge = ≥ 15% / 🟠 orange = 5-15% / 🟢 vert = < 5%")

Nat_orig_s = df["Nature"].fillna("__VIDE__").astype(str)
Nat_pred_s = df["Nature_predite"].fillna("__VIDE__").astype(str)
df_loc = df.copy()
df_loc["est_modifie"] = Nat_orig_s != Nat_pred_s

order_ven_vol = df_loc["Vendeur"].value_counts().index.tolist()
ven_corr = df_loc.groupby("Vendeur").agg(
    n_lignes=("Libelle", "size"),
    n_modifies=("est_modifie", "sum"),
).reindex(order_ven_vol).reset_index()
ven_corr["taux_pct"] = ven_corr["n_modifies"] / ven_corr["n_lignes"] * 100
ven_corr["color"] = ven_corr["taux_pct"].apply(
    lambda t: "#d62728" if t >= 15 else "#ff7f0e" if t >= 5 else "#2ca02c")

fig = go.Figure()
fig.add_trace(go.Bar(
    y=ven_corr["Vendeur"], x=ven_corr["taux_pct"], orientation="h",
    marker_color=ven_corr["color"],
    text=[f"{t:.2f}%  ({m:,} / {n:,})" for t, m, n in zip(
        ven_corr["taux_pct"], ven_corr["n_modifies"], ven_corr["n_lignes"])],
    textposition="outside",
))
fig.update_layout(height=400, xaxis_title="Taux de correction (%)",
                   yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# Total
n_tot = len(df_loc); n_mod = int(df_loc["est_modifie"].sum())
st.caption(f"**Total dataset** : {n_mod:,} / {n_tot:,} ({n_mod/n_tot*100:.2f}%)")

# ============================================================
# Helper : stacked bar 100%
# ============================================================
def tracer_barres_empilees_100(piv, order_cols, order_rows, colors, title):
    sub = piv.reindex(index=order_rows, columns=order_cols).fillna(0).astype(int)
    totals = sub.sum(axis=1)
    pct = sub.div(totals, axis=0) * 100
    fig = go.Figure()
    for col, color in zip(order_cols, colors):
        fig.add_trace(go.Bar(
            y=[f"{v} ({totals[v]:,})" for v in order_rows],
            x=pct[col].values, orientation="h",
            name=col, marker_color=color,
            text=[f"{nb:,} ({w:.1f}%)" if w >= 3.5 else (f"{nb:,}" if w >= 0.5 else "")
                  for nb, w in zip(sub[col].values, pct[col].values)],
            textposition="inside", insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack", height=70 * len(order_rows) + 100,
        xaxis_title="Répartition (%)",
        yaxis=dict(autorange="reversed"),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. Répartition Couleur par Vendeur
# ============================================================
st.subheader("4. Répartition Couleur_Commentaire par Vendeur")
if {"Vendeur", "Couleur_Commentaire"}.issubset(df.columns):
    order_couleur = ["Aucune couleur", "Couleur unique", "2 couleurs", "3 couleurs",
                      "4 couleurs", "5 couleurs", "6 couleurs", "7 couleurs", "8 couleurs"]
    colors_couleur = ["#d62728", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00",
                       "#a65628", "#f781bf", "#999999", "#666666"]
    couleur_piv = pd.crosstab(df["Vendeur"], df["Couleur_Commentaire"])
    tracer_barres_empilees_100(
        couleur_piv,
        [c for c in order_couleur if c in couleur_piv.columns],
        order_ven_vol,
        colors_couleur[:sum(c in couleur_piv.columns for c in order_couleur)],
        "Couleurs détectées par Vendeur (rouge = aucune couleur, signal métier)",
    )
else:
    st.info("Lance l'**extraction couleur** (page « 🎨 Extraction couleur ») pour afficher cette section.")

# ============================================================
# 5. Répartition Dimension par Vendeur
# ============================================================
st.subheader("5. Répartition Dimension_Commentaire par Vendeur")
if {"Vendeur", "Dimension_Commentaire"}.issubset(df.columns):
    order_dim = ["Aucune dimension", "1 dimension", "2 dimensions", "3 dimensions",
                  "Diamètre", "Avec alerte"]
    colors_dim = ["#cccccc", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#d62728"]
    dim_piv = pd.crosstab(df["Vendeur"], df["Dimension_Commentaire"])
    tracer_barres_empilees_100(
        dim_piv,
        [c for c in order_dim if c in dim_piv.columns],
        order_ven_vol,
        colors_dim[:sum(c in dim_piv.columns for c in order_dim)],
        "Dimensions détectées par Vendeur",
    )
else:
    st.info("Lance l'**extraction dimension** (page « 📐 Extraction dimension ») pour afficher cette section.")

# ============================================================
# Tableau de synthese
# ============================================================
st.subheader("Tableau de synthèse")
synthese = pd.DataFrame({
    "Vendeur": ven_corr["Vendeur"],
    "Lignes": ven_corr["n_lignes"],
    "Modifiées Nature": ven_corr["n_modifies"],
    "Taux %": ven_corr["taux_pct"].round(2),
})
st.dataframe(synthese, use_container_width=True, hide_index=True)
