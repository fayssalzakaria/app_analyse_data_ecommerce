"""Page 3 - Recatégorisation de la Nature, calculée à la demande sur le fichier chargé.

Le modèle (TF-IDF + régression logistique, voir core/recat.py) est entraîné ici même
quand l'utilisateur clique sur le bouton. Il produit :
- la Nature prédite (Nature_predite),
- un score de confiance (Nature_Score),
- un commentaire en 7 tranches (Garde / Modifie x niveau de confiance).

Une option « passe 2 » (self-training) est disponible.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import appliquer_filtres_globaux, afficher_filtres_sidebar
from core import enrich


st.set_page_config(page_title="Algo categorisation Nature", page_icon="🎯", layout="wide")
st.title("🎯 Algo de catégorisation Nature (Pass 1 + Pass 2)")

st.markdown(
    """
L'**algo de catégorisation Nature** est calculé **en direct** sur le fichier chargé :

- un modèle **TF-IDF (mots + caractères)** est entraîné sur les libellés dont la
  `Nature` est connue (pondéré par fréquence, + Vendeur et prix si disponibles) ;
- il prédit une `Nature_predite` avec un **score de confiance** par ligne ;
- **cascade** : si le score ≥ seuil → on applique la prédiction (et on remplit les
  `Nature` vides) ; sinon on conserve la `Nature` d'origine.
"""
)

df_full = enrich.obtenir_df_travail()
if df_full is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

with st.expander("⚙️ Paramètres de recatégorisation",
                 expanded=not enrich.possede_colonnes(["Nature_predite"])):
    two_pass = st.checkbox(
        "Activer la passe 2 (self-training)",
        value=False,
        help="La passe 2 ré-entraîne après avoir ré-étiqueté la cible des lignes que la passe 1 "
             "prédit autrement avec confiance. Cascade : score Pass 2 prioritaire, repli sur Pass 1.",
    )
    s1, s2 = st.columns(2)
    seuil_p2 = s1.slider("Seuil Pass 2", 0.30, 0.95, 0.50, 0.05,
                         help="Score Pass 2 ≥ seuil → on applique la prédiction Pass 2.")
    seuil_p1 = s2.slider("Seuil Pass 1 (repli)", 0.30, 0.95, 0.80, 0.05,
                         disabled=not two_pass,
                         help="En 2 passes : si Pass 2 < son seuil mais Pass 1 ≥ ce seuil → on applique Pass 1.")
    cc1, cc2 = st.columns(2)
    use_v = cc1.checkbox("Utiliser le Vendeur", value="Vendeur" in df_full.columns,
                         disabled="Vendeur" not in df_full.columns)
    use_p = cc2.checkbox("Utiliser le prix", value="Montant_cmd" in df_full.columns,
                         disabled="Montant_cmd" not in df_full.columns)
    if st.button("🎯 Lancer la recatégorisation", type="primary"):
        with st.spinner("Entraînement + prédiction en cours..."):
            info = enrich.calculer_recat(two_pass=two_pass, seuil_p2=seuil_p2, seuil_p1=seuil_p1,
                                    use_vendeur=use_v, use_prix=use_p)
        if info.get("ok"):
            msg = (f"Recatégorisation calculée : {info['n_modifies']:,} lignes modifiées "
                   f"({info['pct_modifies']:.2f}%), {info['n_classes']} natures, "
                   f"score médian {info['score_median']:.3f}.")
            if info.get("two_pass"):
                msg += f" Passe 2 : {info.get('n_relabel', 0):,} ligne(s) ré-étiquetée(s) (self-training)."
            st.success(msg)
            st.rerun()
        else:
            st.error(info.get("msg", "Échec du calcul."))

if not enrich.possede_colonnes(["Nature_predite", "Nature_Score", "Nature_Commentaire"]):
    st.info("👆 Lance la recatégorisation pour calculer **Nature_predite** sur le fichier chargé.")
    st.stop()

df_full = enrich.obtenir_df_travail()
afficher_filtres_sidebar(df_full)
df = appliquer_filtres_globaux(df_full)
st.success(f"Dataset enrichi : **{len(df):,}** lignes × **{len(df.columns)}** colonnes.")

# Détail de la dernière recatégorisation (notamment la passe 2)
_rinfo = st.session_state.get("recat_info") or {}
if _rinfo.get("two_pass"):
    st.caption(
        f"🔁 **2 passes** — Pass 2 ≥ {_rinfo.get('seuil_p2')} prioritaire, repli Pass 1 ≥ {_rinfo.get('seuil_p1')}. "
        f"{_rinfo.get('n_relabel', 0):,} ligne(s) ré-étiquetée(s) par self-training."
    )
elif _rinfo:
    st.caption(f"1 passe — seuil {_rinfo.get('seuil_p2')}.")

# ============================================================
# KPIs Nature_predite
# ============================================================
st.subheader("Chiffres-clés")

Nat_orig = df["Nature"].fillna("__VIDE__").astype(str)
Nat_pred = df["Nature_predite"].fillna("__VIDE__").astype(str)
is_modif = Nat_orig != Nat_pred
n_mod = int(is_modif.sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes analysées", f"{len(df):,}")
c2.metric("Lignes Nature modifiée", f"{n_mod:,}", delta=f"{n_mod/len(df)*100:.2f}%")
c3.metric("Score médian", f"{df['Nature_Score'].median():.3f}")
c4.metric("Score ≥ 0.90", f"{(df['Nature_Score'] >= 0.90).sum():,}",
           delta=f"{(df['Nature_Score'] >= 0.90).mean()*100:.1f}% du dataset")

# ============================================================
# Distribution des 7 tranches de commentaire
# ============================================================
st.subheader("Distribution des commentaires (7 tranches)")

order_comm = [
    "Garde - Tres probable", "Garde - Probable", "Garde - Possible", "Garde - Hesitant",
    "Modifie - Tres probable", "Modifie - Probable", "Modifie - Possible", "Modifie - Hesitant",
]
colors_comm = {
    "Garde - Tres probable": "#2ca02c", "Garde - Probable": "#74c476",
    "Garde - Possible": "#bae4b3", "Garde - Hesitant": "#cccccc",
    "Modifie - Tres probable": "#d62728", "Modifie - Probable": "#ff9896",
    "Modifie - Possible": "#ffc1c1", "Modifie - Hesitant": "#ffe6e6",
}

counts = df["Nature_Commentaire"].value_counts().reindex(order_comm).fillna(0).astype(int)
fig = px.bar(
    x=counts.index, y=counts.values,
    color=counts.index, color_discrete_map=colors_comm,
    labels={"x": "Commentaire", "y": "Nombre de lignes"},
    text=[f"{v:,}<br>({v/len(df)*100:.2f}%)" if v > 0 else "" for v in counts.values],
)
fig.update_traces(textposition="outside", showlegend=False)
fig.update_layout(height=500, xaxis_tickangle=-20, margin=dict(t=30, b=80))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Distribution Nature_Score
# ============================================================
st.subheader("Distribution du score de confiance (par action)")
df_plot = df[["Nature_Score"]].copy()
df_plot["Action"] = np.where(is_modif.values, "Modifié", "Gardé")

fig = px.histogram(
    df_plot, x="Nature_Score", color="Action", nbins=50,
    barmode="overlay", opacity=0.7,
    color_discrete_map={"Gardé": "#2ca02c", "Modifié": "#d62728"},
)
fig.add_vline(x=0.50, line_dash="dash", line_color="orange",
               annotation_text="seuil 0.50", annotation_position="top")
fig.add_vline(x=0.70, line_dash="dot", line_color="gray", annotation_text="0.70")
fig.add_vline(x=0.90, line_dash="dot", line_color="gray", annotation_text="0.90")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Top des transitions (avant -> après)
# ============================================================
st.subheader("Top des transitions Nature originale → Nature_predite")
trans = (
    df[is_modif][["Nature", "Nature_predite"]]
    .fillna("__VIDE__")
    .groupby(["Nature", "Nature_predite"]).size()
    .reset_index(name="nb")
    .sort_values("nb", ascending=False)
    .head(20)
)
st.dataframe(trans, use_container_width=True, hide_index=True)

# ============================================================
# Téléchargement rapide du résultat de recatégorisation
# ============================================================
st.subheader("Export")
st.caption("Pour l'export complet (Parquet/CSV/Excel), voir la page « 💾 Export ».")
_csv = df[["Libelle", "Nature", "Nature_predite", "Nature_Score", "Nature_Commentaire"]].to_csv(
    index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("📄 Télécharger les prédictions Nature (CSV)", data=_csv,
                   file_name="recategorisation_nature.csv", mime="text/csv")
