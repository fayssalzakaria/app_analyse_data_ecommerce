"""Page 2 - Visualisation : graphe relationnel, projection 2D/3D, Sankey Univers -> Nature."""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import obtenir_dataframe_actif, afficher_filtres_sidebar, appliquer_filtres_globaux
from core.data_loader import charger_dataset_purifie
from core.embeddings import encoder_textes, reduire_2d, reduire_3d, echantillon_stratifie, HAS_ST, HAS_UMAP
from core.cache_utils import construire_cle_cache, charger_cache_pickle, sauvegarder_cache_pickle


VISUALIZATION_CACHE_VERSION = "visualisation-semantique-2026-05-25-v1"
# Nombre max de couples (Univers / Nature) proposés dans le filtre, classés par fréquence.
MAX_COUPLE_OPTIONS = 30


def _compute_options_for_filters(df: pd.DataFrame) -> dict:
    """Construit les listes d'options pour Tab 4/Tab 5 a partir du df. Resultat memoise par session."""
    cache_token = id(df)
    state_key = f"_viz_options_{cache_token}"
    if state_key in st.session_state:
        return st.session_state[state_key]
    out = {}
    if "Vendeur" in df.columns:
        out["vendeurs"] = sorted({str(v).strip() for v in df["Vendeur"].dropna().unique()})
    else:
        out["vendeurs"] = []
    if "Nature" in df.columns:
        out["natures"] = sorted({str(v).strip() for v in df["Nature"].dropna().unique()})
    else:
        out["natures"] = []
    if "Univers" in df.columns:
        out["univers"] = sorted({str(v).strip() for v in df["Univers"].dropna().unique()})
    else:
        out["univers"] = []
    if {"Univers", "Nature"}.issubset(df.columns):
        sub = df[["Univers", "Nature"]].dropna()
        couples = sub["Univers"].astype(str).str.strip() + " / " + sub["Nature"].astype(str).str.strip()
        out["available_couples"] = set(couples.unique())
        # Couples les plus fréquents en premier (défaut data-driven, indépendant du dataset).
        out["couples_by_freq"] = couples.value_counts().index.tolist()
    else:
        out["available_couples"] = set()
        out["couples_by_freq"] = []
    st.session_state[state_key] = out
    return out


def _scaled_values(values: pd.Series, mode: str, out_min: float, out_max: float) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    if mode == "Lineaire":
        transformed = vals
    elif mode == "Log":
        transformed = np.log1p(vals)
    else:
        transformed = np.sqrt(vals)

    lo = float(transformed.min())
    hi = float(transformed.max())
    if hi <= lo:
        return pd.Series((out_min + out_max) / 2, index=values.index)
    return out_min + (transformed - lo) / (hi - lo) * (out_max - out_min)


def _prepare_relation_graph(
    source_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    min_orders: int,
    max_edges: int,
    multi_mode: str,
    query: str,
    graph_vendeurs: list[str],
    graph_natures: list[str],
    graph_univers: list[str],
    graph_couples: list[str],
    graph_date_range,
    node_metric: str,
    node_scale: str,
    node_min: int,
    node_max: int,
    edge_scale: str,
    edge_min: int,
    edge_max: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    needed_cols = [source_col, target_col, "Cod_cmd"]
    for optional_col in ["Vendeur", "Nature", "Univers", "Date"]:
        if optional_col in source_df.columns and optional_col not in needed_cols:
            needed_cols.append(optional_col)

    cache_key = construire_cle_cache(
        "semantic_relation_graph",
        VISUALIZATION_CACHE_VERSION,
        source_df,
        columns=needed_cols,
        params={
            "source_col": source_col,
            "target_col": target_col,
            "min_orders": min_orders,
            "max_edges": max_edges,
            "multi_mode": multi_mode,
            "query": query,
            "graph_vendeurs": list(graph_vendeurs or []),
            "graph_natures": list(graph_natures or []),
            "graph_univers": list(graph_univers or []),
            "graph_couples": list(graph_couples or []),
            "graph_date_range": str(graph_date_range),
            "node_metric": node_metric,
            "node_scale": node_scale,
            "node_min": node_min,
            "node_max": node_max,
            "edge_scale": edge_scale,
            "edge_min": edge_min,
            "edge_max": edge_max,
        },
    )
    cached = charger_cache_pickle("semantic_relation_graph", cache_key)
    if cached is not None:
        return cached["edges"], cached["nodes"], cached["total_edges"]

    graph_df = source_df[needed_cols].dropna(subset=[source_col, target_col]).copy()
    graph_df[source_col] = graph_df[source_col].astype(str)
    graph_df[target_col] = graph_df[target_col].astype(str)
    for filter_col in ["Vendeur", "Nature", "Univers"]:
        if filter_col in graph_df.columns:
            graph_df[filter_col] = graph_df[filter_col].astype(str).str.strip()

    if graph_vendeurs and "Vendeur" in graph_df.columns:
        graph_df = graph_df[graph_df["Vendeur"].isin(graph_vendeurs)]
    if graph_natures and "Nature" in graph_df.columns:
        graph_df = graph_df[graph_df["Nature"].isin(graph_natures)]
    if graph_univers and "Univers" in graph_df.columns:
        graph_df = graph_df[graph_df["Univers"].isin(graph_univers)]
    if graph_couples and {"Univers", "Nature"}.issubset(graph_df.columns):
        pair = graph_df["Univers"].astype(str).str.strip() + " / " + graph_df["Nature"].astype(str).str.strip()
        graph_df = graph_df[pair.isin(graph_couples)]

    if graph_date_range is not None and "Date" in graph_df.columns:
        if isinstance(graph_date_range, tuple) and len(graph_date_range) == 2:
            start_date, end_date = pd.Timestamp(graph_date_range[0]), pd.Timestamp(graph_date_range[1])
            graph_df = graph_df[(graph_df["Date"] >= start_date) & (graph_df["Date"] <= end_date)]

    if query:
        q = query.lower()
        graph_df = graph_df[
            graph_df[source_col].str.lower().str.contains(q, na=False)
            | graph_df[target_col].str.lower().str.contains(q, na=False)
        ]

    edges = (
        graph_df.groupby([source_col, target_col], dropna=False)
        .agg(nb_lignes=("Cod_cmd", "size"), nb_commandes=("Cod_cmd", "nunique"))
        .reset_index()
    )
    edges = edges[edges["nb_commandes"] >= min_orders]

    if multi_mode == "Source -> plusieurs cibles" and not edges.empty:
        multi_sources = edges.groupby(source_col)[target_col].nunique()
        edges = edges[edges[source_col].isin(multi_sources[multi_sources > 1].index)]
    elif multi_mode == "Cible <- plusieurs sources" and not edges.empty:
        multi_targets = edges.groupby(target_col)[source_col].nunique()
        edges = edges[edges[target_col].isin(multi_targets[multi_targets > 1].index)]

    total_edges = len(edges)
    edges = edges.sort_values("nb_commandes", ascending=False).head(max_edges).copy()

    if edges.empty:
        return edges, pd.DataFrame(), total_edges

    node_source = edges[[source_col, "nb_lignes", "nb_commandes"]].rename(columns={source_col: "value"})
    node_source["type"] = source_col
    node_target = edges[[target_col, "nb_lignes", "nb_commandes"]].rename(columns={target_col: "value"})
    node_target["type"] = target_col
    node_rows = pd.concat([node_source, node_target], ignore_index=True)
    nodes = node_rows.groupby(["type", "value"], as_index=False).agg(
        nb_lignes=("nb_lignes", "sum"),
        nb_commandes=("nb_commandes", "sum"),
    )
    nodes["size_scaled"] = _scaled_values(nodes[node_metric], node_scale, node_min, node_max)
    edges["width_scaled"] = _scaled_values(edges["nb_commandes"], edge_scale, edge_min, edge_max)
    sauvegarder_cache_pickle(
        "semantic_relation_graph",
        cache_key,
        {"edges": edges, "nodes": nodes, "total_edges": total_edges},
    )
    return edges, nodes, total_edges


def _semantic_projection_cache_key(
    data: pd.DataFrame,
    sample_size: int,
    dim_3d: bool,
    color_by: str | None,
) -> str:
    cols = [c for c in ["Libelle", "Nature", "Univers", "Vendeur"] if c in data.columns]
    return construire_cle_cache(
        "semantic_projection",
        VISUALIZATION_CACHE_VERSION,
        data,
        columns=cols,
        params={
            "sample_size": sample_size,
            "dim_3d": bool(dim_3d),
            "color_by": color_by,
            "encoder": "sentence_transformers" if HAS_ST else "tfidf_svd",
            "reducer": "umap" if HAS_UMAP else "svd",
        },
        code_files=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "embeddings.py")],
    )


def _load_or_build_semantic_projection(
    data: pd.DataFrame,
    sample_size: int,
    dim_3d: bool,
    color_by: str | None,
) -> tuple[pd.DataFrame, bool]:
    cache_key = _semantic_projection_cache_key(data, sample_size, dim_3d, color_by)
    cached = charger_cache_pickle("semantic_projection", cache_key)
    if cached is not None:
        return cached, True

    sample = echantillon_stratifie(data, sample_size, by="Nature").reset_index(drop=False)
    texts = sample["Libelle"].fillna("").astype(str).tolist()
    text_hash = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    emb_cache_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "cache",
        f"emb_{len(texts)}_{text_hash}.npy",
    )
    emb = encoder_textes(texts, cache_path=emb_cache_path)
    if dim_3d:
        xy = reduire_3d(emb)
        sample["x"], sample["y"], sample["z"] = xy[:, 0], xy[:, 1], xy[:, 2]
    else:
        xy = reduire_2d(emb)
        sample["x"], sample["y"] = xy[:, 0], xy[:, 1]
    sauvegarder_cache_pickle("semantic_projection", cache_key, sample)
    return sample, False


def _load_or_build_sankey_edges(data: pd.DataFrame, top_n_natures: int) -> tuple[pd.DataFrame, bool]:
    cache_key = construire_cle_cache(
        "semantic_sankey",
        VISUALIZATION_CACHE_VERSION,
        data,
        columns=["Univers", "Nature"],
        params={"top_n_natures": top_n_natures},
    )
    cached = charger_cache_pickle("semantic_sankey", cache_key)
    if cached is not None:
        return cached, True

    g = data.dropna(subset=["Univers", "Nature"]).groupby(["Univers", "Nature"]).size().reset_index(name="nb")
    top_natures = g.groupby("Nature")["nb"].sum().nlargest(top_n_natures).index
    g = g[g["Nature"].isin(top_natures)].copy()
    sauvegarder_cache_pickle("semantic_sankey", cache_key, g)
    return g, False


def _load_or_build_suspect_pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    cache_key = construire_cle_cache(
        "semantic_suspect_pairs",
        VISUALIZATION_CACHE_VERSION,
        data,
        columns=["Univers", "Nature"],
        params={"min_rows": 30, "top": 40},
    )
    cached = charger_cache_pickle("semantic_suspect_pairs", cache_key)
    if cached is not None:
        return cached, True

    g = data.dropna(subset=["Univers", "Nature"]).groupby(["Nature", "Univers"]).size().reset_index(name="nb")
    majo = g.loc[g.groupby("Nature")["nb"].idxmax()][["Nature", "Univers"]].rename(
        columns={"Univers": "Univers_majo"}
    )
    m = g.merge(majo, on="Nature")
    suspects = m[m["Univers"] != m["Univers_majo"]].copy()
    suspects = suspects[suspects["nb"] >= 30].sort_values("nb", ascending=False).head(40)
    sauvegarder_cache_pickle("semantic_suspect_pairs", cache_key, suspects)
    return suspects, False


def _relation_panel_js() -> str:
    return """
    <div id="graph-info-panel" style="
        position:absolute; top:14px; right:14px; z-index:9999;
        width:300px; max-height:245px; overflow:auto;
        background:rgba(255,255,255,0.96); border:1px solid #d8dde8;
        border-radius:8px; padding:12px 14px;
        box-shadow:0 8px 24px rgba(15,23,42,0.16);
        font-family:Arial,sans-serif; font-size:13px; color:#1f2937;">
        <div style="font-weight:700; margin-bottom:6px;">Survoler un noeud ou un lien</div>
        <div style="color:#6b7280;">Le resume apparait ici.</div>
    </div>
    <button id="graph-fullscreen-btn" style="
        position:absolute; top:14px; left:14px; z-index:9999;
        background:#1f2937; color:#fff; border:none; border-radius:6px;
        padding:8px 14px; font-family:Arial,sans-serif; font-size:13px;
        font-weight:600; cursor:pointer; box-shadow:0 4px 12px rgba(15,23,42,0.2);">
        &#9974; Plein ecran
    </button>
    <script>
    (function(){
        var btn = document.getElementById("graph-fullscreen-btn");
        function inFull(){ return document.fullscreenElement || document.webkitFullscreenElement; }
        function req(el){
            if (el.requestFullscreen) return el.requestFullscreen();
            if (el.webkitRequestFullscreen) return el.webkitRequestFullscreen();
        }
        function exit(){
            if (document.exitFullscreen) return document.exitFullscreen();
            if (document.webkitExitFullscreen) return document.webkitExitFullscreen();
        }
        if (btn){
            btn.addEventListener("click", function(){
                if (inFull()) { exit(); } else { req(document.documentElement); }
            });
        }
        function resizeGraph(){
            var net = document.getElementById("mynetwork");
            if (!net) return;
            if (inFull()) {
                net.style.height = "100vh";
                net.style.width = "100vw";
            } else {
                net.style.height = "640px";
                net.style.width = "100%";
            }
            if (typeof network !== "undefined" && network.redraw) {
                try { network.redraw(); network.fit(); } catch(e){}
            }
        }
        document.addEventListener("fullscreenchange", function(){
            if (btn) btn.innerHTML = inFull() ? "&#9974; Quitter" : "&#9974; Plein ecran";
            resizeGraph();
        });
        document.addEventListener("webkitfullscreenchange", function(){
            if (btn) btn.innerHTML = inFull() ? "&#9974; Quitter" : "&#9974; Plein ecran";
            resizeGraph();
        });
    })();
    </script>
    <script>
    (function(){
        function esc(v){
            return String(v ?? "").replace(/[&<>"']/g, function(c){
                if (c === "&") return "&amp;";
                if (c === "<") return "&lt;";
                if (c === ">") return "&gt;";
                if (c === '"') return "&quot;";
                return "&#39;";
            });
        }
        function fmt(v){ return Number(v || 0).toLocaleString("fr-FR"); }
        function setPanel(html){
            var p = document.getElementById("graph-info-panel");
            if (p) p.innerHTML = html;
        }
        function nodeHtml(node){
            return "<div style='font-weight:700; margin-bottom:8px;'>Noeud</div>"
                + "<div><b>Type :</b> " + esc(node.node_type || node.group) + "</div>"
                + "<div><b>Nom :</b> " + esc(node.full_label || node.label) + "</div>"
                + "<div><b>Lignes :</b> " + fmt(node.nb_lignes) + "</div>"
                + "<div><b>Commandes :</b> " + fmt(node.nb_commandes) + "</div>";
        }
        function edgeHtml(edge){
            return "<div style='font-weight:700; margin-bottom:8px;'>Lien</div>"
                + "<div><b>Origine :</b> " + esc(edge.source_type) + " - " + esc(edge.source_label) + "</div>"
                + "<div><b>Cible :</b> " + esc(edge.target_type) + " - " + esc(edge.target_label) + "</div>"
                + "<div><b>Commandes :</b> " + fmt(edge.nb_commandes) + "</div>"
                + "<div><b>Lignes :</b> " + fmt(edge.nb_lignes) + "</div>";
        }
        var tries = 0;
        var physicsStopped = false;
        function stopPhysics(){
            if (physicsStopped) return;
            if (typeof network !== "undefined" && network.setOptions) {
                try { network.setOptions({ physics: { enabled: false } }); physicsStopped = true; } catch(e){}
            }
        }
        var timer = setInterval(function(){
            tries++;
            if (typeof network !== "undefined" && typeof nodes !== "undefined" && typeof edges !== "undefined") {
                clearInterval(timer);
                // Borne le moteur physique a 200 iterations puis le coupe.
                try {
                    network.setOptions({
                        physics: {
                            stabilization: { enabled: true, iterations: 200, updateInterval: 25, fit: true },
                            adaptiveTimestep: true,
                            timestep: 0.5
                        }
                    });
                } catch(e){}
                network.once("stabilizationIterationsDone", stopPhysics);
                network.once("stabilized", stopPhysics);
                // Filet de securite : kill physics apres 6s si rien ne s'est stabilise.
                setTimeout(stopPhysics, 6000);

                network.on("hoverNode", function(params){
                    var node = nodes.get(params.node);
                    if (node) setPanel(nodeHtml(node));
                });
                network.on("hoverEdge", function(params){
                    var edge = edges.get(params.edge);
                    if (edge) setPanel(edgeHtml(edge));
                });
                network.on("selectNode", function(params){
                    var node = nodes.get(params.nodes[0]);
                    if (node) setPanel(nodeHtml(node));
                });
                network.on("selectEdge", function(params){
                    var edge = edges.get(params.edges[0]);
                    if (edge) setPanel(edgeHtml(edge));
                });
            }
            if (tries > 50) clearInterval(timer);
        }, 100);
    })();
    </script>
    """


def _render_relation_graph(
    title: str,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    total_edges: int,
    source_col: str,
    target_col: str,
    pair_label: str,
) -> None:
    st.markdown(f"**{title}**")
    if edges.empty:
        st.info("Aucun lien a afficher avec ces filtres.")
        return

    st.caption(f"{len(edges):,} liens affiches sur {total_edges:,}. {len(nodes):,} noeuds.")

    if HAS_PYVIS:
        net = Network(height="640px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        net.barnes_hut(gravity=-25000, central_gravity=0.25, spring_length=180, spring_strength=0.02)
        colors = {"Libelle": "#4C78A8", "Nature": "#F58518", "Univers": "#54A24B"}

        for row in nodes.itertuples(index=False):
            node_id = f"{row.type}::{row.value}"
            net.add_node(
                node_id,
                label=str(row.value)[:55],
                title=f"{row.type}: {row.value}<br>Lignes: {int(row.nb_lignes):,}<br>Commandes: {int(row.nb_commandes):,}",
                size=float(row.size_scaled),
                color=colors.get(row.type, "#777777"),
                group=row.type,
                node_type=row.type,
                full_label=str(row.value),
                nb_lignes=int(row.nb_lignes),
                nb_commandes=int(row.nb_commandes),
            )

        for row in edges.itertuples(index=False):
            src_value = getattr(row, source_col)
            tgt_value = getattr(row, target_col)
            src = f"{source_col}::{src_value}"
            tgt = f"{target_col}::{tgt_value}"
            nb_cmd = int(row.nb_commandes)
            nb_lignes = int(row.nb_lignes)
            net.add_edge(
                src,
                tgt,
                id=f"{src}-->{tgt}",
                value=nb_cmd,
                width=float(row.width_scaled),
                label=str(nb_cmd),
                title=f"{src_value} -> {tgt_value}<br>Commandes: {nb_cmd:,}<br>Lignes: {nb_lignes:,}",
                source_label=str(src_value),
                target_label=str(tgt_value),
                source_type=source_col,
                target_type=target_col,
                nb_commandes=nb_cmd,
                nb_lignes=nb_lignes,
            )

        html = net.generate_html(notebook=False).replace("</body>", _relation_panel_js() + "</body>")
        components.html(html, height=820, scrolling=True)
    else:
        st.warning("pyvis indisponible : affichage fallback Sankey.")
        labels = list(pd.unique(pd.concat([edges[source_col], edges[target_col]], ignore_index=True)))
        label_idx = {label: i for i, label in enumerate(labels)}
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=10, thickness=14),
            link=dict(
                source=edges[source_col].map(label_idx),
                target=edges[target_col].map(label_idx),
                value=edges["nb_commandes"],
                label=edges["nb_commandes"].astype(str),
            ),
        ))
        fig.update_layout(height=640, title=f"Graphe agrege {pair_label} - {title}")
        st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title="Visualisation graphe", page_icon="🌐", layout="wide")
st.title("🌐 Visualisation graphe")

df = obtenir_dataframe_actif()
if df is None:
    st.warning("Aucun dataset charge.")
    st.stop()

if "dataset_purifie" not in st.session_state:
    purified_from_disk = charger_dataset_purifie()
    if purified_from_disk is not None:
        st.session_state["dataset_purifie"] = purified_from_disk
        st.session_state["dataset_purifie_ready"] = True

use_clean_available = st.session_state.get("dataset_purifie_ready") and "dataset_purifie" in st.session_state
source_options = ["Dataset brut"]
if use_clean_available:
    source_options.append("Dataset purifie")
_src_default = st.session_state.get("viz_source_dataset", "Dataset brut")
source_dataset = st.radio(
    "Source de visualisation",
    source_options,
    index=source_options.index(_src_default) if _src_default in source_options else 0,
    horizontal=True,
    help="Le dataset purifie est disponible apres passage par la page 7 Export.",
)
st.session_state["viz_source_dataset"] = source_dataset
if source_dataset == "Dataset purifie":
    df = st.session_state["dataset_purifie"]
    st.success("Visualisation branchee sur le dataset purifie persistant.")
elif not use_clean_available:
    st.info("Passe par la page 7 Export pour rendre le dataset purifie disponible ici.")

afficher_filtres_sidebar(df)
df = appliquer_filtres_globaux(df)

if not HAS_ST:
    st.caption("Mode rapide : encodage des libelles avec TF-IDF + SVD.")
if not HAS_UMAP:
    st.caption("Mode rapide : reduction de dimension avec SVD.")

tab_graphe, tab_nuage, tab_sankey, tab_anormaux, tab_avant = st.tabs(
    ["Graphe relationnel", "Nuage de points", "Sankey", "Liens anormaux", "Avant/apres"]
)

# ============ Nuage de points (UMAP) ============
with tab_nuage:
    st.subheader("Projection 2D/3D des libelles")
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_size = st.slider("Taille echantillon", 2000, 20000, 8000, step=1000)
    with c2:
        color_options = [c for c in ["Univers", "Nature", "Vendeur"] if c in df.columns]
        color_by = st.selectbox("Colorer par", color_options, index=0) if color_options else None
    with c3:
        dim_3d = st.checkbox("3D au lieu de 2D", value=False)

    if st.button("Charger / calculer la projection", type="primary"):
        with st.spinner("Chargement depuis cache disque ou calcul de la projection..."):
            sample, from_cache = _load_or_build_semantic_projection(df, sample_size, dim_3d, color_by)
        st.session_state["umap_sample"] = sample
        st.session_state["umap_3d"] = dim_3d
        st.session_state["umap_color_by"] = color_by
        if from_cache:
            st.success("Projection rechargee depuis le cache disque.")
        else:
            st.success("Projection calculee et sauvegardee dans le cache disque.")

    if "umap_sample" in st.session_state:
        sample = st.session_state["umap_sample"]
        is_3d = st.session_state.get("umap_3d", False)
        cby = st.session_state.get("umap_color_by", "Univers")
        cby = cby if cby in sample.columns else None
        hover_cols = [c for c in ["Libelle", "Nature", "Univers", "Vendeur"] if c in sample.columns]
        if is_3d:
            fig = px.scatter_3d(
                sample, x="x", y="y", z="z",
                color=cby, hover_data=hover_cols,
                opacity=0.6,
            )
        else:
            fig = px.scatter(
                sample, x="x", y="y",
                color=cby, hover_data=hover_cols,
                opacity=0.5,
            )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=700, legend=dict(itemsizing="constant"))
        st.plotly_chart(fig, use_container_width=True)

# ============ Sankey ============
with tab_sankey:
    st.subheader("Sankey diagram - Univers -> Nature")
    if {"Univers", "Nature"}.issubset(df.columns):
        top_n_natures = st.slider("Top N natures a afficher", 10, 60, 30)
        g, sankey_from_cache = _load_or_build_sankey_edges(df, top_n_natures)
        if sankey_from_cache:
            st.caption("Sankey recharge depuis le cache disque.")

        labels = list(g["Univers"].unique()) + list(g["Nature"].unique())
        label_idx = {l: i for i, l in enumerate(labels)}
        src = g["Univers"].map(label_idx).tolist()
        tgt = g["Nature"].map(label_idx).tolist()
        val = g["nb"].tolist()

        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=10, thickness=14),
            link=dict(source=src, target=tgt, value=val),
        ))
        fig.update_layout(height=800, title="Flux des lignes : Univers -> Nature (top N natures)")
        st.plotly_chart(fig, use_container_width=True)

# ============ Liens anormaux ============
with tab_anormaux:
    st.subheader("Paires (Univers, Nature) suspectes")
    st.caption(
        "On considere qu'une Nature appartient principalement a un Univers : "
        "l'Univers ou elle apparait le plus. Les paires (Univers, Nature) "
        "ou cette Nature n'est PAS dans son Univers majoritaire sont suspectes."
    )
    if {"Univers", "Nature"}.issubset(df.columns):
        suspects, suspects_from_cache = _load_or_build_suspect_pairs(df)
        if suspects_from_cache:
            st.caption("Paires suspectes rechargees depuis le cache disque.")
        st.dataframe(
            suspects[["Nature", "Univers", "Univers_majo", "nb"]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Lecture : pour la Nature X, son Univers majoritaire est *Univers_majo* "
                   "mais on a *nb* lignes dans un autre Univers - cas typique d'import vendeur defectueux.")


# ============ Graphe relationnel (onglet par defaut) ============
with tab_graphe:
    st.subheader("Graphe relationnel agrege")
    st.caption(
        "Chaque noeud represente une valeur de Libelle, Nature ou Univers. "
        "La taille des noeuds depend du nombre de lignes associees. "
        "La taille et le libelle des liens indiquent le nombre de commandes uniques."
    )

    pair_options = {
        "Libelle -> Nature": ("Libelle", "Nature"),
        "Libelle -> Univers": ("Libelle", "Univers"),
        "Univers -> Nature": ("Univers", "Nature"),
    }
    available_pairs = {
        label: cols for label, cols in pair_options.items()
        if set(cols).issubset(df.columns)
    }

    if not available_pairs:
        st.info("Colonnes insuffisantes pour construire le graphe relationnel.")
    else:
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            pair_label = st.selectbox("Couple a afficher", list(available_pairs.keys()))
            source_col, target_col = available_pairs[pair_label]
        with gc2:
            min_orders = st.number_input("Commandes min. par lien", min_value=1, max_value=10000, value=1, step=1)
        with gc3:
            max_edges = st.slider("Nombre max. de liens", 20, 500, 120, step=20)
        with gc4:
            _mm_opts = ["Tous", "Source -> plusieurs cibles", "Cible <- plusieurs sources"]
            _mm_default = st.session_state.get("viz_multi_mode", "Source -> plusieurs cibles")
            multi_mode = st.selectbox(
                "Filtre liens multiples",
                _mm_opts,
                index=_mm_opts.index(_mm_default) if _mm_default in _mm_opts else 0,
            )
            st.session_state["viz_multi_mode"] = multi_mode

        query = st.text_input(
            "Filtrer une valeur (libelle, Nature ou Univers)",
            placeholder="ex. un libellé, une Nature ou un Univers...",
        )

        _viz_opts = _compute_options_for_filters(df)
        gf1, gf2, gf3 = st.columns(3)
        with gf1:
            if _viz_opts["vendeurs"]:
                _vd_default = st.session_state.get("viz_graph_vendeurs", [])
                graph_vendeurs = st.multiselect(
                    "Filtre vendeur",
                    _viz_opts["vendeurs"],
                    default=[v for v in _vd_default if v in _viz_opts["vendeurs"]],
                    help="Vide = tous les vendeurs.",
                )
                st.session_state["viz_graph_vendeurs"] = graph_vendeurs
            else:
                graph_vendeurs = []
        with gf2:
            if _viz_opts["natures"]:
                graph_natures = st.multiselect(
                    "Filtre Nature",
                    _viz_opts["natures"],
                    default=[],
                    help="Vide = toutes les natures.",
                )
            else:
                graph_natures = []
        with gf3:
            if _viz_opts["univers"]:
                graph_univers = st.multiselect(
                    "Filtre Univers",
                    _viz_opts["univers"],
                    default=[],
                    help="Vide = tous les univers.",
                )
            else:
                graph_univers = []

        gf4, gf5 = st.columns(2)
        with gf4:
            if _viz_opts["available_couples"]:
                couple_options = _viz_opts["couples_by_freq"][:MAX_COUPLE_OPTIONS]
                graph_couples = st.multiselect(
                    "Filtre couple Univers / Nature",
                    couple_options,
                    default=[],
                    help="Vide = pas de filtre couple. Couples les plus fréquents proposés en premier.",
                )
            else:
                graph_couples = []
        with gf5:
            if "Date" in df.columns and df["Date"].notna().any():
                dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
                graph_date_range = st.date_input(
                    "Filtre periode",
                    value=(dmin, dmax),
                    min_value=dmin,
                    max_value=dmax,
                )
            else:
                graph_date_range = None

        with st.expander("Reglage des tailles", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                node_metric = st.selectbox("Taille des noeuds basee sur", ["nb_lignes", "nb_commandes"], index=0)
                node_scale = st.selectbox("Echelle noeuds", ["Racine", "Log", "Lineaire"], index=0)
            with sc2:
                node_min = st.slider("Taille noeud min", 4, 40, 10)
                node_max = st.slider("Taille noeud max", 20, 100, 55)
            with sc3:
                edge_scale = st.selectbox("Echelle liens", ["Racine", "Log", "Lineaire"], index=0)
                edge_min = st.slider("Epaisseur lien min", 1, 10, 1)
                edge_max = st.slider("Epaisseur lien max", 2, 25, 12)
            node_max = max(node_max, node_min + 1)
            edge_max = max(edge_max, edge_min + 1)

        edges, nodes, total_edges = _prepare_relation_graph(
            df,
            source_col=source_col,
            target_col=target_col,
            min_orders=min_orders,
            max_edges=max_edges,
            multi_mode=multi_mode,
            query=query,
            graph_vendeurs=graph_vendeurs,
            graph_natures=graph_natures,
            graph_univers=graph_univers,
            graph_couples=graph_couples,
            graph_date_range=graph_date_range,
            node_metric=node_metric,
            node_scale=node_scale,
            node_min=node_min,
            node_max=node_max,
            edge_scale=edge_scale,
            edge_min=edge_min,
            edge_max=edge_max,
        )
        _render_relation_graph("Graphe relationnel", edges, nodes, total_edges, source_col, target_col, pair_label)

        if not edges.empty:
            with st.expander("Voir les liens agreges"):
                st.dataframe(
                    edges.rename(columns={source_col: "source", target_col: "cible"}),
                    use_container_width=True,
                    hide_index=True,
                )


# ============ Avant / apres correction ============
with tab_avant:
    st.subheader("Comparaison graphe avant / apres correction")
    st.caption(
        "Les deux graphes utilisent les memes filtres et les memes reglages. "
        "Le graphe apres correction remplace uniquement la colonne Nature par Nature_finale."
    )

    have_recat = "recat_corrections" in st.session_state
    if not have_recat:
        st.warning("Lance d'abord la page 3 pour calculer la correction Nature.")
    elif "Nature" not in df.columns:
        st.warning("La colonne Nature est absente du dataset courant.")
    else:
        corrections = st.session_state["recat_corrections"]
        df_after = df.copy()
        common_idx = df_after.index.intersection(corrections.index)
        df_after.loc[common_idx, "Nature"] = corrections.loc[common_idx, "Nature_finale"].values

        pair_options = {
            "Libelle -> Nature": ("Libelle", "Nature"),
            "Libelle -> Univers": ("Libelle", "Univers"),
            "Univers -> Nature": ("Univers", "Nature"),
        }
        available_pairs = {
            label: cols for label, cols in pair_options.items()
            if set(cols).issubset(df.columns)
        }

        if not available_pairs:
            st.info("Colonnes insuffisantes pour construire la comparaison.")
        else:
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                pair_label_cmp = st.selectbox("Couple compare", list(available_pairs.keys()), key="cmp_pair")
                source_cmp, target_cmp = available_pairs[pair_label_cmp]
            with cc2:
                min_orders_cmp = st.number_input(
                    "Commandes min. par lien",
                    min_value=1,
                    max_value=10000,
                    value=1,
                    step=1,
                    key="cmp_min_orders",
                )
            with cc3:
                max_edges_cmp = st.slider("Nombre max. de liens", 20, 500, 120, step=20, key="cmp_max_edges")
            with cc4:
                multi_mode_cmp = st.selectbox(
                    "Filtre liens multiples",
                    ["Tous", "Source -> plusieurs cibles", "Cible <- plusieurs sources"],
                    key="cmp_multi_mode",
                )

            query_cmp = st.text_input(
                "Filtrer une valeur commune",
                placeholder="ex. un libellé, une Nature ou un Univers...",
                key="cmp_query",
            )

            cf1, cf2, cf3 = st.columns(3)
            with cf1:
                if "Vendeur" in df.columns:
                    vendeur_options_cmp = sorted([str(v).strip() for v in df["Vendeur"].dropna().unique()])
                    graph_vendeurs_cmp = st.multiselect(
                        "Filtre vendeur commun",
                        vendeur_options_cmp,
                        default=[],
                        help="Vide = tous les vendeurs.",
                        key="cmp_vendeurs",
                    )
                else:
                    graph_vendeurs_cmp = []
            with cf2:
                if "Nature" in df.columns:
                    nature_options_cmp = sorted([str(v).strip() for v in df["Nature"].dropna().unique()])
                    graph_natures_cmp = st.multiselect(
                        "Filtre Nature commun",
                        nature_options_cmp,
                        default=[],
                        help="Vide = toutes les natures.",
                        key="cmp_natures",
                    )
                else:
                    graph_natures_cmp = []
            with cf3:
                if "Univers" in df.columns:
                    univers_options_cmp = sorted([str(v).strip() for v in df["Univers"].dropna().unique()])
                    graph_univers_cmp = st.multiselect(
                        "Filtre Univers commun",
                        univers_options_cmp,
                        default=[],
                        help="Vide = tous les univers.",
                        key="cmp_univers",
                    )
                else:
                    graph_univers_cmp = []

            cf4, cf5 = st.columns(2)
            with cf4:
                if {"Univers", "Nature"}.issubset(df.columns):
                    _couples_cmp = (
                        df["Univers"].fillna("").astype(str).str.strip()
                        + " / "
                        + df["Nature"].fillna("").astype(str).str.strip()
                    )
                    _couples_cmp = _couples_cmp[_couples_cmp.str.strip(" /") != ""]
                    couple_options_cmp = _couples_cmp.value_counts().index.tolist()[:MAX_COUPLE_OPTIONS]
                    graph_couples_cmp = st.multiselect(
                        "Filtre couple Univers / Nature (commun)",
                        couple_options_cmp,
                        default=[],
                        help="Vide = pas de filtre couple.",
                        key="cmp_sensitive_couples",
                    )
                else:
                    graph_couples_cmp = []
            with cf5:
                if "Date" in df.columns and df["Date"].notna().any():
                    dmin_cmp, dmax_cmp = df["Date"].min().date(), df["Date"].max().date()
                    graph_date_range_cmp = st.date_input(
                        "Filtre periode commun",
                        value=(dmin_cmp, dmax_cmp),
                        min_value=dmin_cmp,
                        max_value=dmax_cmp,
                        key="cmp_date_range",
                    )
                else:
                    graph_date_range_cmp = None

            with st.expander("Reglage commun des tailles", expanded=False):
                cs1, cs2, cs3 = st.columns(3)
                with cs1:
                    node_metric_cmp = st.selectbox(
                        "Taille des noeuds basee sur",
                        ["nb_lignes", "nb_commandes"],
                        index=0,
                        key="cmp_node_metric",
                    )
                    node_scale_cmp = st.selectbox("Echelle noeuds", ["Racine", "Log", "Lineaire"], index=0, key="cmp_node_scale")
                with cs2:
                    node_min_cmp = st.slider("Taille noeud min", 4, 40, 10, key="cmp_node_min")
                    node_max_cmp = st.slider("Taille noeud max", 20, 100, 55, key="cmp_node_max")
                with cs3:
                    edge_scale_cmp = st.selectbox("Echelle liens", ["Racine", "Log", "Lineaire"], index=0, key="cmp_edge_scale")
                    edge_min_cmp = st.slider("Epaisseur lien min", 1, 10, 1, key="cmp_edge_min")
                    edge_max_cmp = st.slider("Epaisseur lien max", 2, 25, 12, key="cmp_edge_max")
                node_max_cmp = max(node_max_cmp, node_min_cmp + 1)
                edge_max_cmp = max(edge_max_cmp, edge_min_cmp + 1)

            args_common = dict(
                source_col=source_cmp,
                target_col=target_cmp,
                min_orders=min_orders_cmp,
                max_edges=max_edges_cmp,
                multi_mode=multi_mode_cmp,
                query=query_cmp,
                graph_vendeurs=graph_vendeurs_cmp,
                graph_natures=graph_natures_cmp,
                graph_univers=graph_univers_cmp,
                graph_couples=graph_couples_cmp,
                graph_date_range=graph_date_range_cmp,
                node_metric=node_metric_cmp,
                node_scale=node_scale_cmp,
                node_min=node_min_cmp,
                node_max=node_max_cmp,
                edge_scale=edge_scale_cmp,
                edge_min=edge_min_cmp,
                edge_max=edge_max_cmp,
            )
            edges_before, nodes_before, total_before = _prepare_relation_graph(df, **args_common)
            edges_after, nodes_after, total_after = _prepare_relation_graph(df_after, **args_common)

            left, right = st.columns(2)
            with left:
                _render_relation_graph(
                    "Avant correction",
                    edges_before,
                    nodes_before,
                    total_before,
                    source_cmp,
                    target_cmp,
                    pair_label_cmp,
                )
            with right:
                _render_relation_graph(
                    "Apres correction",
                    edges_after,
                    nodes_after,
                    total_after,
                    source_cmp,
                    target_cmp,
                    pair_label_cmp,
                )

            with st.expander("Comparer les liens agreges"):
                b = edges_before.rename(columns={source_cmp: "source", target_cmp: "cible"}).copy()
                a = edges_after.rename(columns={source_cmp: "source", target_cmp: "cible"}).copy()
                b["etat"] = "Avant"
                a["etat"] = "Apres"
                st.dataframe(
                    pd.concat([b, a], ignore_index=True),
                    use_container_width=True,
                    hide_index=True,
                )
