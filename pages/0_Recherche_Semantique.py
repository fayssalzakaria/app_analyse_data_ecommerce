"""Page 0 - Recherche multi-critere + recherche semantique (aide a la saisie)."""
import os
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import afficher_filtres_sidebar, appliquer_filtres_globaux
from core import enrich
from core.extract import extraire_couleur_contextuelle, extraire_dimension
from core.search import construire_catalogue_libelles, construire_index_tfidf, rechercher_similaires, suggerer_categorie
from core.cache_utils import construire_cle_cache, charger_cache_pickle, sauvegarder_cache_pickle


SEARCH_CACHE_VERSION = "recherche-semantique-2026-05-25-v1"
SEARCH_CORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core",
    "search.py",
)


st.set_page_config(page_title="Recherche semantique", page_icon="🔎", layout="wide")
st.title("🔎 Recherche semantique & aide a la saisie")

df_base = enrich.obtenir_df_travail()
if df_base is None:
    st.warning("Aucun dataset charge.")
    st.stop()

# Recherche possible sur le dataset brut (Nature d'origine) OU sur la version recategorisee
# (Nature_predite), si la recategorisation a ete lancee dans l'app.
has_recat = enrich.possede_colonnes(["Nature_predite"])
source_dataset = st.radio(
    "Source de recherche",
    ["Dataset brut", "Dataset corrige (recategorise)"],
    index=0,
    horizontal=True,
    help="Brut = Nature d'origine. Corrige = Nature recategorisee (Nature_predite).",
)

if source_dataset == "Dataset corrige (recategorise)":
    if not has_recat:
        st.info(
            "Veuillez d'abord lancer la **recategorisation** sur la page "
            "« 🎯 Algo categorisation Nature » pour activer cette source."
        )
        st.stop()
    df = df_base.copy()
    df["Nature"] = df["Nature_predite"].fillna(df["Nature"])  # recherche sur la categorie corrigee
    st.success("Recherche branchee sur les Natures recategorisees (Nature_predite).")
else:
    df = df_base.copy()

afficher_filtres_sidebar(df)
df = appliquer_filtres_globaux(df)
is_clean_source = source_dataset != "Dataset brut"

# ============================================================
# Recherche multi-critere : une seule barre en langage naturel.
# Le parseur separe le texte (RAG) des criteres (prix / vendeur / couleur / dimension).
# ============================================================
st.subheader("🔎 Recherche")
st.caption(
    "Langage naturel : un nom de produit + des critères. "
    "Ex : `prix 19` · `couleur rouge 80x150` · `hauteur 10` · `prix > 100` · `vendeur X`."
)
st.caption(
    "💡 Combine texte libre et critères (prix, vendeur, couleur, dimension). "
    "Bascule entre _Dataset brut_ et _Dataset corrigé_ pour comparer l'avant/après."
)

_DIM_KEYWORDS = {
    "hauteur": "H_cm", "haut": "H_cm",
    "largeur": "l_cm", "larg": "l_cm",
    "longueur": "L_cm", "long": "L_cm", "profondeur": "L_cm", "prof": "L_cm",
    "diametre": "diametre_cm", "diamètre": "diametre_cm", "diam": "diametre_cm",
}
_NUM = r"(\d+(?:[.,]\d+)?)"
_OP = r"(>=|<=|>|<|=)?"


def _num_clause(text, kw_pattern):
    m = re.search(rf"(?:{kw_pattern})\s*{_OP}\s*{_NUM}", text, flags=re.I)
    if not m:
        return None
    return (m.group(1) or "~"), float(m.group(2).replace(",", ".")), m.span()


def _mask_num(series, op, val, tol_abs=None, tol_rel=0.15):
    s = pd.to_numeric(series, errors="coerce")
    if op == ">":
        return s > val
    if op == "<":
        return s < val
    if op == ">=":
        return s >= val
    if op == "<=":
        return s <= val
    if op == "=":
        return s == val
    if tol_abs is not None:
        return (s - val).abs() <= tol_abs
    t = max(val * tol_rel, 1.0)
    return s.between(val - t, val + t)


def analyser_requete(q: str) -> dict:
    text = " " + (q or "") + " "
    crit = {"prix": None, "vendeur": None, "couleur": None, "dims": {}, "dim_label": None}
    chips = []

    pc = _num_clause(text, r"prix|euros?|€")
    if pc:
        op, val, span = pc
        crit["prix"] = (op, val)
        text = text[:span[0]] + " " + text[span[1]:]
        chips.append(f"💶 prix {'≈ ' if op == '~' else op + ' '}{val:g}")

    mv = re.search(r"vendeur\s*(\d+)", text, flags=re.I)
    if mv:
        crit["vendeur"] = f"Vendeur {mv.group(1)}"
        text = text[:mv.start()] + " " + text[mv.end():]
        chips.append(f"🏷️ {crit['vendeur']}")

    for kw, col in _DIM_KEYWORDS.items():
        dc = _num_clause(text, kw)
        if dc:
            op, val, span = dc
            crit["dims"][col] = (op, val)
            text = text[:span[0]] + " " + text[span[1]:]
            chips.append(f"📐 {kw} {'= ' if op == '~' else op + ' '}{val:g} cm")

    mb = re.search(rf"{_NUM}\s*x\s*{_NUM}(?:\s*x\s*{_NUM})?", text, flags=re.I)
    if mb:
        crit["dim_label"] = mb.group(0).replace(" ", "").replace(",", ".").lower()
        text = text[:mb.start()] + " " + text[mb.end():]
        chips.append(f"📐 {crit['dim_label']}")

    try:
        cinfo = extraire_couleur_contextuelle(text, fine=True)
        couleur = cinfo.get("couleur_niveau3") or cinfo.get("couleur_principale")
    except Exception:
        couleur = None
    if couleur:
        crit["couleur"] = str(couleur).strip().lower()
        text = re.sub(re.escape(str(couleur)), " ", text, flags=re.I)
        chips.append(f"🎨 couleur = {crit['couleur']}")

    text = re.sub(r"\b(couleur|coloris|teinte|prix|vendeur|de|en|la|le|les|un|une)\b", " ", text, flags=re.I)
    crit["text"] = re.sub(r"\s+", " ", text).strip()
    if crit["text"]:
        chips.insert(0, f"🔤 « {crit['text']} »")
    crit["chips"] = chips
    return crit


def _mc_full_index(data: pd.DataFrame, source_name: str):
    """Catalogue leger (libelle + frequence) + index TF-IDF, construit UNE fois et cache disque.

    Evite de relancer construire_catalogue_libelles (groupby+mode sur tout le catalogue) a chaque requete.
    """
    ck = construire_cle_cache("mc_search_index", "mc-idx-v2", data, columns=["Libelle"],
                         params={"source": source_name})
    cached = charger_cache_pickle("mc_search_index", ck)
    if cached is not None:
        return cached["catalog"], cached["vectorizer"], cached["matrix"]
    s = data["Libelle"].fillna("").astype(str).str.strip()
    s = s[s != ""]
    vc = s.value_counts()
    cat = pd.DataFrame({"Libelle": vc.index, "nb_lignes": vc.values})
    vect, mat = construire_index_tfidf(cat, mode="Libelle seul")
    sauvegarder_cache_pickle("mc_search_index", ck, {"catalog": cat, "vectorizer": vect, "matrix": mat})
    return cat, vect, mat


def executer_recherche_multicritere(data: pd.DataFrame, crit: dict, source_name: str, max_results: int = 60):
    cat, vect, matrix = _mc_full_index(data, source_name)
    cat = cat.copy()
    libs_cat = cat["Libelle"].fillna("").astype(str).str.strip()
    if crit["text"]:
        from sklearn.metrics.pairwise import cosine_similarity
        qv = vect.transform([crit["text"]])
        cat["score"] = cosine_similarity(qv, matrix).ravel()
    else:
        cat["score"] = float("nan")
    score_map = pd.Series(cat["score"].values, index=libs_cat.values)

    notes = []
    mask = pd.Series(True, index=data.index)
    used = False
    price_active = bool(crit["prix"]) and "Montant_cmd" in data.columns
    if crit["couleur"]:
        if "couleur_extraite" in data.columns:
            mask &= data["couleur_extraite"].fillna("").astype(str).str.lower() == crit["couleur"]
            used = True
        else:
            notes.append("Couleur ignorée : source sans colonne couleur → choisis « Dataset corrigé ».")
    if price_active:
        op, val = crit["prix"]
        mask &= _mask_num(data["Montant_cmd"], op, val, tol_rel=0.15)
        used = True
    if crit["vendeur"] and "Vendeur" in data.columns:
        mask &= data["Vendeur"].fillna("").astype(str).str.strip() == crit["vendeur"]
        used = True
    for col, (op, val) in crit["dims"].items():
        if col in data.columns:
            mask &= _mask_num(data[col], op, val, tol_abs=2.0)
            used = True
        else:
            notes.append(f"{col} ignorée : source sans dimensions → choisis « Dataset corrigé ».")
    if crit["dim_label"] and "dim_label" in data.columns:
        mask &= data["dim_label"].fillna("").astype(str).str.replace(" ", "", regex=False).str.lower() == crit["dim_label"]
        used = True

    sub_n = int(mask.sum()) if used else len(data)

    if price_active:
        # Prix = clef d'agregation : une ligne par (libelle, prix arrondi a l'euro),
        # comptee UNIQUEMENT sur les lignes qui matchent les criteres (nb_lignes honnete).
        sub = data.loc[mask, ["Libelle", "Montant_cmd"]].copy()
        sub["Libelle"] = sub["Libelle"].fillna("").astype(str).str.strip()
        sub = sub[sub["Libelle"] != ""]
        sub["prix"] = pd.to_numeric(sub["Montant_cmd"], errors="coerce").round(0)
        sub = sub.dropna(subset=["prix"])
        res = sub.groupby(["Libelle", "prix"], sort=False).size().reset_index(name="nb_lignes")
        res["prix"] = res["prix"].astype(int)
        res["score"] = res["Libelle"].map(score_map)
    else:
        res = cat.copy()
        if used:
            counts = (data.loc[mask, "Libelle"].fillna("").astype(str).str.strip()
                      .replace("", pd.NA).dropna().value_counts())
            res = res[libs_cat.isin(counts.index)].copy()
            res["nb_lignes"] = libs_cat[res.index].map(counts).astype("Int64")

    if crit["text"]:
        res = res[res["score"] > 0].sort_values(["score", "nb_lignes"], ascending=[False, False])
    else:
        res = res.sort_values("nb_lignes", ascending=False)
    res = res.head(max_results).copy()

    if not res.empty:
        keys = res["Libelle"].astype(str).str.strip()
        tops = set(keys)
        d2 = data[data["Libelle"].astype(str).str.strip().isin(tops)].copy()
        d2["_l"] = d2["Libelle"].astype(str).str.strip()

        def _dominant(col):
            v = d2[col].fillna("").astype(str)
            return d2.assign(_v=v).groupby("_l")["_v"].agg(
                lambda x: x[x != ""].mode().iloc[0] if (x != "").any() else "")

        if "Nature" in data.columns:
            res["Nature"] = keys.map(_dominant("Nature"))
        if "dim_label" in data.columns:
            res["dimension"] = keys.map(_dominant("dim_label"))
        if "Univers" in data.columns:
            res["Univers"] = keys.map(_dominant("Univers"))
        if "Vendeur" in data.columns:
            res["Vendeur"] = keys.map(_dominant("Vendeur"))
    if "score" in res.columns:
        res["score"] = res["score"].round(3)
    return sub_n, res, notes


# Champ de recherche agrandi (lisibilite demo) : CSS limite au formulaire de recherche.
st.markdown(
    """
    <style>
    section[data-testid="stForm"] div[data-testid="stTextInput"] input {
        font-size: 1.5rem !important;
        height: 3.2rem !important;
        padding: 0.4rem 1rem !important;
    }
    section[data-testid="stForm"] div[data-testid="stTextInput"] label p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }
    section[data-testid="stForm"] button[kind="primaryFormSubmit"] {
        font-size: 1.2rem !important;
        height: 3rem !important;
        padding: 0 1.6rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_top_lib = (str(df["Libelle"].dropna().astype(str).value_counts().index[0])
            if "Libelle" in df.columns and df["Libelle"].notna().any() else "")
st.session_state.setdefault("mc_query_input", "")
with st.form("mc_search_form"):
    mc_query = st.text_input(
        "Recherche",
        key="mc_query_input",
        placeholder=(f"ex : {_top_lib}  ·  prix > 100  ·  couleur rouge"
                     if _top_lib else "ex : nom de produit  ·  prix > 100  ·  couleur rouge"),
    )
    mc_submit = st.form_submit_button("Rechercher", type="primary")
if mc_submit:
    st.session_state["mc_query"] = mc_query

_mc_q = st.session_state.get("mc_query", "")
if _mc_q.strip():
    _crit = analyser_requete(_mc_q)
    st.markdown("**Compris :** " + ("  ·  ".join(_crit["chips"]) if _crit["chips"] else "_rien détecté_"))
    _subn, _res, _notes = executer_recherche_multicritere(df, _crit, source_dataset)
    for _n in _notes:
        st.warning(_n)
    _n_lib = int(_res["Libelle"].nunique()) if not _res.empty else 0
    if "prix" in _res.columns:
        st.caption(f"{_subn:,} lignes correspondent · {_n_lib:,} libellés · {len(_res):,} couples (libellé, prix).")
    else:
        st.caption(f"{_subn:,} lignes correspondent aux critères · {_n_lib:,} libellés.")
    _cols = [c for c in ["score", "prix", "Libelle", "dimension", "Univers", "Vendeur", "nb_lignes", "Nature"] if c in _res.columns]
    if not _res.empty and _cols:
        st.dataframe(_res[_cols], use_container_width=True, hide_index=True, height=460)
    else:
        st.info("Aucun résultat pour ces critères.")

st.divider()
if not st.checkbox(
    "Afficher la recherche sémantique avancée (voisins + suggestion — calcul plus lourd, ~50 s la 1re fois)",
    value=False,
):
    st.stop()

with st.expander("Methode", expanded=False):
    st.markdown(
        """
Cette page est un bonus d'aide a la saisie et a l'audit.

Elle agrege les libelles uniques, construit un index TF-IDF leger sur CPU, puis retrouve
les libelles existants les plus proches d'une recherche. Les voisins servent ensuite a
proposer une `Nature` et un `Univers` probables.

La V1 utilise TF-IDF caracteres 3-5 grammes : c'est rapide, explicable et sans nouveau modele.
Une V2 pourra ajouter des embeddings pre-entraines avec cache disque.
        """
    )


def _load_or_build_catalog(data: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, bool]:
    cols = [c for c in ["Libelle", "Nature", "Univers", "Vendeur"] if c in data.columns]
    cache_key = construire_cle_cache(
        "semantic_search_catalog",
        SEARCH_CACHE_VERSION,
        data,
        columns=cols,
        params={"source": source_name},
        code_files=[SEARCH_CORE_PATH, __file__],
    )
    cached = charger_cache_pickle("semantic_search_catalog", cache_key)
    if cached is not None:
        return cached, True
    catalog_built = construire_catalogue_libelles(data)
    sauvegarder_cache_pickle("semantic_search_catalog", cache_key, catalog_built)
    return catalog_built, False


def _load_or_build_index(catalog_data: pd.DataFrame, mode: str, source_name: str):
    cache_key = construire_cle_cache(
        "semantic_search_index",
        SEARCH_CACHE_VERSION,
        catalog_data,
        columns=list(catalog_data.columns),
        params={"source": source_name, "mode": mode},
        code_files=[SEARCH_CORE_PATH, __file__],
    )
    cached = charger_cache_pickle("semantic_search_index", cache_key)
    if cached is not None:
        return cached["vectorizer"], cached["matrix"], True
    vectorizer_built, matrix_built = construire_index_tfidf(catalog_data, mode=mode)
    sauvegarder_cache_pickle(
        "semantic_search_index",
        cache_key,
        {"vectorizer": vectorizer_built, "matrix": matrix_built},
    )
    return vectorizer_built, matrix_built, False


catalog, catalog_from_cache = _load_or_build_catalog(df, source_dataset)
if catalog.empty:
    st.warning("Aucun libelle exploitable pour construire le catalogue.")
    st.stop()
if catalog_from_cache:
    st.caption("Catalogue de recherche recharge depuis le cache disque.")

c1, c2, c3 = st.columns(3)
c1.metric("Libelles uniques", f"{len(catalog):,}")
c2.metric("Lignes filtrees", f"{len(df):,}")
c3.metric("Top libelle occurrences", f"{int(catalog['nb_lignes'].max()):,}")

mode = st.radio(
    "Mode de recherche",
    ["Libelle seul", "Libelle + Univers + Nature"],
    horizontal=True,
)
top_k = st.slider("Nombre de voisins", 5, 100, 20, step=5)

_ex_lib = (str(catalog.sort_values("nb_lignes", ascending=False)["Libelle"].iloc[0])
           if len(catalog) else "nom de produit")
query = st.text_input(
    "Libelle a rechercher ou a saisir",
    value="",
    placeholder=f"ex : {_ex_lib}",
)

with st.spinner("Construction / chargement de l'index de recherche..."):
    vectorizer, matrix, index_from_cache = _load_or_build_index(catalog, mode, source_dataset)
if index_from_cache:
    st.caption("Index TF-IDF recharge depuis le cache disque.")

_go = st.button("Rechercher les libelles similaires", type="primary")
if (_go or query) and query.strip():
    neighbors = rechercher_similaires(query, catalog, vectorizer, matrix, top_k=top_k)
    suggestion = suggerer_categorie(neighbors, top_n=top_k)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Nature recommandee", suggestion["Nature_recommandee"] or "n/a")
    s2.metric("Univers recommande", suggestion["Univers_recommande"] or "n/a")
    s3.metric("Confiance", suggestion["confiance"])
    s4.metric("Score moyen top 5", f"{suggestion['score_moyen_top5']:.3f}")

    color_info = extraire_couleur_contextuelle(query, fine=True)
    dim_info = extraire_dimension(query) or {}
    with st.expander("Extraction attributs du libelle saisi", expanded=True):
        attr = pd.DataFrame([
            {
                "couleur": color_info.get("couleur_niveau3"),
                "finition": color_info.get("finition_niveau3"),
                "decision_couleur": color_info.get("decision_niveau3"),
                "dimension": dim_info.get("dim_label"),
                "L_cm": dim_info.get("L_cm"),
                "l_cm": dim_info.get("l_cm"),
                "H_cm": dim_info.get("H_cm"),
            }
        ])
        st.dataframe(attr, use_container_width=True, hide_index=True)

    if neighbors.empty:
        st.info("Aucun voisin trouve.")
        st.stop()

    st.subheader("Top libelles similaires")
    display_cols = [
        "score",
        "Libelle",
        "nb_lignes",
        "Nature_majoritaire",
        "Univers_majoritaire",
        "Vendeur_majoritaire",
    ]
    display_cols = [c for c in display_cols if c in neighbors.columns]
    neighbors_display = neighbors[display_cols].copy()
    neighbors_display["score"] = neighbors_display["score"].round(3)
    st.dataframe(neighbors_display, use_container_width=True, hide_index=True, height=520)

    st.subheader("Details des lignes du dataset")
    st.caption(
        "Cette vue reutilise le dataset deja charge en session. Elle ne relance pas les algos. "
        "Si la source est le dataset purifie, les colonnes corrigees et les attributs extraits sont affichables ici."
    )

    neighbor_labels = neighbors["Libelle"].dropna().astype(str).tolist()
    selected_labels = st.multiselect(
        "Libelles a inspecter",
        neighbor_labels,
        default=neighbor_labels[: min(3, len(neighbor_labels))],
    )

    if selected_labels:
        detail_rows = df[df["Libelle"].fillna("").astype(str).isin(selected_labels)].copy()
        preferred_cols = [
            "Cod_cmd",
            "Date",
            "Vendeur",
            "Libelle",
            "Univers",
            "Nature",
            "Nature_originale",
            "Nature_corrigee",
            "correction_statut",
            "correction_score",
            "couleur_extraite",
            "dim_label",
            "L_cm",
            "l_cm",
            "H_cm",
            "unite_source",
            "Quantite",
            "Montant_cmd",
            "CA",
        ]
        default_cols = [c for c in preferred_cols if c in detail_rows.columns]
        remaining_cols = [c for c in detail_rows.columns if c not in default_cols]

        dc1, dc2, dc3 = st.columns([2, 1, 1])
        with dc1:
            selected_cols = st.multiselect(
                "Colonnes a afficher",
                default_cols + remaining_cols,
                default=default_cols or list(detail_rows.columns[:12]),
            )
        with dc2:
            max_detail_rows = st.slider("Nombre max. de lignes", 50, 5000, 500, step=50)
        with dc3:
            only_corrected = st.checkbox(
                "Seulement lignes corrigees",
                value=False,
                disabled=not is_clean_source or "correction_statut" not in detail_rows.columns,
            )

        if only_corrected and "correction_statut" in detail_rows.columns:
            detail_rows = detail_rows[detail_rows["correction_statut"].isin(["corrigee", "remplie", "a_revoir"])]

        if selected_cols:
            st.dataframe(
                detail_rows[selected_cols].head(max_detail_rows),
                use_container_width=True,
                hide_index=True,
                height=720,
            )
            st.caption(f"{min(len(detail_rows), max_detail_rows):,} lignes affichees sur {len(detail_rows):,}.")
        else:
            st.info("Selectionne au moins une colonne a afficher.")
    else:
        st.info("Selectionne un libelle voisin pour afficher les lignes detaillees.")

    if "Nature_majoritaire" in neighbors.columns:
        nature_counts = (
            neighbors.groupby("Nature_majoritaire")["score"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="score_cumule")
        )
        fig = px.bar(nature_counts, x="score_cumule", y="Nature_majoritaire", orientation="h", text_auto=True)
        fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    if "Univers_majoritaire" in neighbors.columns:
        univers_counts = (
            neighbors.groupby("Univers_majoritaire")["score"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="score_cumule")
        )
        fig = px.bar(univers_counts, x="score_cumule", y="Univers_majoritaire", orientation="h", text_auto=True)
        fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
