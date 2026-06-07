"""Page 3 - Recatégorisation (Nature ou Univers), calculée à la demande.

Le modèle (TF-IDF + régression logistique, voir core/recat.py) est entraîné ici
même au clic. On peut fiabiliser :
- la Nature (modèle sur le libellé, option passe 2 / self-training),
- l'Univers, déduit de la Nature corrigée pour rester cohérent avec la hiérarchie
  (un Univers contient plusieurs Natures), avec repli modèle si la Nature manque.

Colonnes produites : <cible>_predite, <cible>_Score, <cible>_Commentaire.
"""
import os
import sys

import numpy as np
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import appliquer_filtres_globaux, afficher_filtres_sidebar
from core import enrich


st.set_page_config(page_title="Categorisation Nature / Univers", page_icon="🎯", layout="wide")
st.title("🎯 Catégorisation Nature / Univers")

df_full = enrich.obtenir_df_travail()
if df_full is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

# Choix de la catégorie à fiabiliser. L'Univers dépend de la Nature (voir plus bas).
cible = st.radio(
    "Catégorie à fiabiliser",
    ["Nature", "Univers"],
    horizontal=True,
    help="L'Univers est déduit de la Nature corrigée : lance d'abord la Nature.",
)
col_pred = f"{cible}_predite"
col_score = f"{cible}_Score"
col_comm = f"{cible}_Commentaire"

st.markdown(
    f"""
La catégorisation est calculée **en direct** sur le fichier chargé.

- **Nature** : un modèle **TF-IDF (mots + caractères)** est entraîné sur les libellés
  dont la `Nature` est connue (+ Vendeur et prix si dispo), puis prédit `Nature_predite`
  avec un score. Option **passe 2** (self-training).
- **Univers** : on apprend la table `Nature → Univers majoritaire` et on attribue à
  chaque ligne l'Univers de sa Nature corrigée (**cohérence garantie**) ; repli sur un
  modèle libellé→Univers pour les lignes sans Nature.

Cible courante : **{cible}**.
"""
)

# ------------------------------------------------------------------
# Paramètres + bouton de calcul (dépendent de la cible)
# ------------------------------------------------------------------
with st.expander("⚙️ Paramètres", expanded=not enrich.possede_colonnes([col_pred])):
    if cible == "Nature":
        two_pass = st.checkbox(
            "Activer la passe 2 (self-training)",
            value=False,
            help="Ré-entraîne après avoir ré-étiqueté les lignes que la passe 1 prédit "
                 "autrement avec confiance. Cascade : score Pass 2 prioritaire, repli Pass 1.",
        )
        s1, s2 = st.columns(2)
        seuil_p2 = s1.slider("Seuil Pass 2", 0.30, 0.95, 0.50, 0.05)
        seuil_p1 = s2.slider("Seuil Pass 1 (repli)", 0.30, 0.95, 0.80, 0.05, disabled=not two_pass)
        cc1, cc2 = st.columns(2)
        use_v = cc1.checkbox("Utiliser le Vendeur", value="Vendeur" in df_full.columns,
                             disabled="Vendeur" not in df_full.columns)
        use_p = cc2.checkbox("Utiliser le prix", value="Montant_cmd" in df_full.columns,
                             disabled="Montant_cmd" not in df_full.columns)
        if st.button("🎯 Lancer la recatégorisation Nature", type="primary"):
            with st.spinner("Entraînement + prédiction en cours..."):
                info = enrich.calculer_recat(cible="Nature", two_pass=two_pass,
                                             seuil_p2=seuil_p2, seuil_p1=seuil_p1,
                                             use_vendeur=use_v, use_prix=use_p)
            if info.get("ok"):
                msg = (f"Nature calculée : {info['n_modifies']:,} lignes modifiées "
                       f"({info['pct_modifies']:.2f}%), {info['n_classes']} natures, "
                       f"score médian {info['score_median']:.3f}.")
                if info.get("two_pass"):
                    msg += f" Passe 2 : {info.get('n_relabel', 0):,} ligne(s) ré-étiquetée(s)."
                st.success(msg)
                st.rerun()
            else:
                st.error(info.get("msg", "Échec du calcul."))
    else:  # Univers
        st.caption(
            "L'Univers est déduit de la **Nature corrigée** (cohérence garantie) "
            "+ repli sur un modèle libellé→Univers pour les lignes sans Nature."
        )
        nature_prete = enrich.possede_colonnes(["Nature_predite"])
        if not nature_prete:
            st.warning("Lance d'abord la recatégorisation **Nature** (choisis « Nature » ci-dessus).")
        if st.button("🌍 Calculer l'Univers", type="primary", disabled=not nature_prete):
            with st.spinner("Calcul de l'Univers (table Nature→Univers + repli modèle)..."):
                info = enrich.calculer_univers()
            if info.get("ok"):
                st.success(
                    f"Univers calculé : {info['n_modifies']:,} lignes modifiées "
                    f"({info['pct_modifies']:.2f}%). {info['n_par_table']:,} par la table Nature→Univers, "
                    f"{info['n_par_modele']:,} par le modèle de repli."
                )
                st.rerun()
            else:
                st.error(info.get("msg", "Échec du calcul."))

if not enrich.possede_colonnes([col_pred, col_score, col_comm]):
    st.info(f"👆 Lance le calcul pour fiabiliser **{cible}** sur le fichier chargé.")
    st.stop()

df_full = enrich.obtenir_df_travail()
afficher_filtres_sidebar(df_full)
df = appliquer_filtres_globaux(df_full)
st.success(f"Dataset enrichi : **{len(df):,}** lignes × **{len(df.columns)}** colonnes.")

# Rappel du paramétrage du dernier calcul.
_rinfo = st.session_state.get(f"recat_info_{cible}") or {}
if cible == "Nature" and _rinfo.get("two_pass"):
    st.caption(f"🔁 2 passes — Pass 2 ≥ {_rinfo.get('seuil_p2')}, repli Pass 1 ≥ {_rinfo.get('seuil_p1')} "
               f"({_rinfo.get('n_relabel', 0):,} lignes ré-étiquetées).")
elif cible == "Univers" and _rinfo:
    st.caption(f"🌍 {_rinfo.get('n_natures_mappees', 0)} Natures mappées vers un Univers "
               f"(cohérence garantie avec la Nature corrigée).")

# ============================================================
# Chiffres-clés
# ============================================================
st.subheader("Chiffres-clés")
orig_str = df[cible].fillna("__VIDE__").astype(str)
pred_str = df[col_pred].fillna("__VIDE__").astype(str)
is_modif = orig_str != pred_str
n_mod = int(is_modif.sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes analysées", f"{len(df):,}")
c2.metric(f"Lignes {cible} modifiée", f"{n_mod:,}", delta=f"{n_mod/len(df)*100:.2f}%")
c3.metric("Score médian", f"{df[col_score].median():.3f}")
c4.metric("Score ≥ 0.90", f"{(df[col_score] >= 0.90).sum():,}",
           delta=f"{(df[col_score] >= 0.90).mean()*100:.1f}% du dataset")

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
counts = df[col_comm].value_counts().reindex(order_comm).fillna(0).astype(int)
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
# Distribution du score de confiance (par action)
# ============================================================
st.subheader("Distribution du score de confiance (par action)")
df_plot = df[[col_score]].copy()
df_plot["Action"] = np.where(is_modif.values, "Modifié", "Gardé")
fig = px.histogram(
    df_plot, x=col_score, color="Action", nbins=50,
    barmode="overlay", opacity=0.7,
    color_discrete_map={"Gardé": "#2ca02c", "Modifié": "#d62728"},
)
fig.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="0.50", annotation_position="top")
fig.add_vline(x=0.90, line_dash="dot", line_color="gray", annotation_text="0.90")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Top des transitions (avant -> après)
# ============================================================
st.subheader(f"Top des transitions {cible} originale → {col_pred}")
trans = (
    df[is_modif][[cible, col_pred]]
    .fillna("__VIDE__")
    .groupby([cible, col_pred]).size()
    .reset_index(name="nb")
    .sort_values("nb", ascending=False)
    .head(20)
)
st.dataframe(trans, use_container_width=True, hide_index=True)

# ============================================================
# Téléchargement rapide du résultat
# ============================================================
st.subheader("Export")
st.caption("Pour l'export complet (Parquet/CSV/Excel), voir la page « 💾 Export ».")
_cols = [c for c in ["Libelle", cible, col_pred, col_score, col_comm] if c in df.columns]
_csv = df[_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(f"📄 Télécharger les prédictions {cible} (CSV)", data=_csv,
                   file_name=f"recategorisation_{cible.lower()}.csv", mime="text/csv")
