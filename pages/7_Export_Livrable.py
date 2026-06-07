"""Page 7 - Export du dataset enrichi (CSV / Excel / Parquet)."""
import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import (
    LIVRABLE_FINAL_PATH,
    appliquer_filtres_globaux,
    afficher_filtres_sidebar,
)
from core import enrich


st.set_page_config(page_title="Export livrable", page_icon="💾", layout="wide")
st.title("💾 Export du dataset enrichi")

df = enrich.obtenir_df_travail()
if df is None:
    st.warning("Aucun dataset chargé. Retourne à l'accueil pour en charger un.")
    st.stop()

_enriched = [c for c in (enrich.RECAT_COLS + enrich.UNIVERS_COLS + enrich.COLOR_COLS + enrich.DIM_COLS)
             if c in df.columns]
if _enriched:
    st.caption("Colonnes enrichies présentes : " + ", ".join(f"`{c}`" for c in _enriched))
else:
    st.info("Aucun enrichissement calculé pour l'instant — l'export contiendra le dataset brut. "
            "Lance la recatégorisation / couleur / dimension sur les pages dédiées pour enrichir.")

afficher_filtres_sidebar(df)
df_filtered = appliquer_filtres_globaux(df)

# ============================================================
# Apercu
# ============================================================
n_total = len(df)
n_filtered = len(df_filtered)
c1, c2, c3 = st.columns(3)
c1.metric("Lignes (dataset complet)", f"{n_total:,}")
c2.metric("Lignes (après filtre sidebar)", f"{n_filtered:,}")
c3.metric("Colonnes", f"{len(df.columns)}")

st.subheader("Aperçu (30 premières lignes)")
st.dataframe(df_filtered.head(30), use_container_width=True, hide_index=True)

# ============================================================
# Structure des colonnes
# ============================================================
with st.expander("Structure des colonnes", expanded=False):
    info = []
    for c in df.columns:
        info.append({
            "Colonne": c,
            "Type": str(df[c].dtype),
            "Non-nuls": int(df[c].notna().sum()),
            "Uniques": int(df[c].nunique(dropna=True)),
        })
    st.dataframe(pd.DataFrame(info), use_container_width=True, hide_index=True)

# ============================================================
# Telechargements
# ============================================================
st.subheader("Téléchargements")
st.caption("Téléchargements basés sur le filtre sidebar courant. Sans filtre = livrable complet.")

c1, c2, c3 = st.columns(3)

with c1:
    # parquet (rapide)
    buf = io.BytesIO()
    df_filtered.to_parquet(buf, index=False)
    st.download_button(
        label="📦 Parquet",
        data=buf.getvalue(),
        file_name="dataset_enrichi.parquet",
        mime="application/octet-stream",
        help="Format compact pour réutilisation programmatique",
    )

with c2:
    # CSV (compatible Excel)
    csv_bytes = df_filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📄 CSV",
        data=csv_bytes,
        file_name="dataset_enrichi.csv",
        mime="text/csv",
        help="UTF-8 BOM, ouvrable dans Excel",
    )

with c3:
    # Excel : genere a la demande (lent sur de gros volumes), conserve en session_state
    # pour que le bouton de telechargement reste fiable apres le rerun.
    if st.button("📊 Préparer Excel (~1-2 min)"):
        with st.spinner("Génération du fichier Excel..."):
            xbuf = io.BytesIO()
            df_filtered.to_excel(xbuf, sheet_name="Predictions", index=False, engine="openpyxl")
            st.session_state["export_xlsx_bytes"] = xbuf.getvalue()
            st.session_state["export_xlsx_rows"] = len(df_filtered)
    if st.session_state.get("export_xlsx_bytes"):
        st.download_button(
            label="⬇️ Excel (.xlsx)",
            data=st.session_state["export_xlsx_bytes"],
            file_name="dataset_enrichi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Prêt ({st.session_state.get('export_xlsx_rows', 0):,} lignes). "
                 "Reclique « Préparer » si tu changes le filtre.",
        )

# ============================================================
# Info source
# ============================================================
with st.expander("Source des données", expanded=False):
    if os.path.exists(LIVRABLE_FINAL_PATH):
        _size = os.path.getsize(LIVRABLE_FINAL_PATH) / 1024 / 1024
        st.caption(f"Livrable pré-calculé détecté dans `data/livrables/` ({_size:.1f} MB).")
    else:
        st.caption("Aucun livrable pré-calculé : le dataset est enrichi à la volée dans l'app.")
