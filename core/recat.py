"""Recatégorisation Nature — version générique calculée à la volée.

Algo (1 passe, faiblement supervisé) inspiré du pipeline v3g :
- TF-IDF mots (1-2 grammes) + TF-IDF caractères (char_wb 3-5) sur le libellé,
- + dummies vendeur (top-K, le reste regroupé) si dispo,
- + statistiques de prix par libellé (min/médiane/max du log-prix) si dispo,
- LogisticRegression entraînée sur les lignes dont la Nature est connue
  (agrégées par (libellé, Nature, vendeur) avec poids = nb d'occurrences).

Aucune valeur n'est codée en dur : tout est dérivé du dataset fourni.
Sortie : Nature_predite, Nature_Score, Nature_Commentaire (7 tranches).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Seuils de la cascade. En 1 passe, seul DEFAULT_SEUIL_P2 sert.
# En 2 passes : score Pass 2 ≥ seuil_p2 -> Pass 2 ; sinon score Pass 1 ≥ seuil_p1 -> Pass 1 ;
# sinon on conserve la Nature d'origine.
DEFAULT_SEUIL_P2 = 0.50
DEFAULT_SEUIL_P1 = 0.80
DEFAULT_SEUIL = DEFAULT_SEUIL_P2  # rétro-compat
VENDEUR_TOP_K = 8
RECAT_VERSION = "recat-generic-v2"


def _norm_lib(s) -> str | None:
    if pd.isna(s):
        return None
    return " ".join(unicodedata.normalize("NFC", str(s).strip().lower()).split())


def _vendeur_groupe(vendeur: pd.Series, top_k: int = VENDEUR_TOP_K) -> pd.Series:
    """Garde les top_k vendeurs les plus fréquents, regroupe le reste sous '__autre__'."""
    v = vendeur.fillna("__na__").astype(str)
    top = v.value_counts().head(top_k).index
    return v.where(v.isin(top), "__autre__")


def _log_prix(df: pd.DataFrame) -> pd.Series | None:
    if {"Montant_cmd", "Quantite"}.issubset(df.columns):
        qte = pd.to_numeric(df["Quantite"], errors="coerce").replace(0, 1)
        prix_unit = pd.to_numeric(df["Montant_cmd"], errors="coerce") / qte
        return np.log1p(prix_unit.fillna(0).clip(lower=0))
    if "Montant_cmd" in df.columns:
        return np.log1p(pd.to_numeric(df["Montant_cmd"], errors="coerce").fillna(0).clip(lower=0))
    return None


def _build_features_fit(g: pd.DataFrame, use_vendeur: bool, use_prix: bool):
    vect_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5,
                             max_features=15000, sublinear_tf=True)
    Xw = vect_w.fit_transform(g["_lib_n"])
    vect_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.5,
                             max_features=15000, sublinear_tf=True)
    Xc = vect_c.fit_transform(g["_lib_n"])
    blocks = [Xw, Xc]

    ven_cols: list[str] = []
    if use_vendeur and "Vendeur_grp" in g.columns:
        ven_d = pd.get_dummies(g["Vendeur_grp"], prefix="vend").astype(np.float32)
        ven_cols = ven_d.columns.tolist()
        blocks.append(csr_matrix(ven_d.values))

    prix_mean = prix_std = None
    if use_prix and {"lp_min", "lp_med", "lp_max"}.issubset(g.columns):
        prix_arr = g[["lp_min", "lp_med", "lp_max"]].fillna(0).values.astype(np.float32)
        prix_mean = prix_arr.mean(axis=0)
        prix_std = prix_arr.std(axis=0).clip(min=1e-6)
        blocks.append(csr_matrix((prix_arr - prix_mean) / prix_std))

    X = hstack(blocks).tocsr()
    X.sort_indices(); X.sum_duplicates()
    bundle = {"vect_w": vect_w, "vect_c": vect_c, "ven_cols": ven_cols,
              "prix_mean": prix_mean, "prix_std": prix_std}
    return X, bundle


def _build_features_transform(df_p: pd.DataFrame, bundle: dict):
    Xw = bundle["vect_w"].transform(df_p["_lib_n"].fillna(""))
    Xc = bundle["vect_c"].transform(df_p["_lib_n"].fillna(""))
    blocks = [Xw, Xc]
    if bundle["ven_cols"]:
        ven = pd.get_dummies(df_p["Vendeur_grp"], prefix="vend").reindex(
            columns=bundle["ven_cols"], fill_value=0).astype(np.float32)
        blocks.append(csr_matrix(ven.values))
    if bundle["prix_mean"] is not None:
        prix_arr = df_p[["lp_min", "lp_med", "lp_max"]].fillna(0).values.astype(np.float32)
        blocks.append(csr_matrix((prix_arr - bundle["prix_mean"]) / bundle["prix_std"]))
    return hstack(blocks).tocsr()


def entrainer_recat(df: pd.DataFrame, use_vendeur: bool = True, use_prix: bool = True) -> dict | None:
    """Entraîne le modèle sur les lignes dont la Nature est connue. None si pas assez de données."""
    if "Libelle" not in df.columns or "Nature" not in df.columns:
        return None
    work = pd.DataFrame({"_lib_n": df["Libelle"].map(_norm_lib),
                         "target": df["Nature"]})
    if use_vendeur and "Vendeur" in df.columns:
        work["Vendeur_grp"] = _vendeur_groupe(df["Vendeur"]).values
    lp = _log_prix(df) if use_prix else None
    if lp is not None:
        work["log_prix"] = lp.values

    train = work[work["_lib_n"].notna() & work["target"].notna()].copy()
    if train["target"].nunique() < 2 or len(train) < 20:
        return None  # pas assez de signal pour entraîner

    group_cols = ["_lib_n", "target"]
    if "Vendeur_grp" in train.columns:
        group_cols.append("Vendeur_grp")
    g = train.groupby(group_cols, dropna=False).size().reset_index(name="weight")

    prix_stats = None
    if "log_prix" in train.columns:
        prix_stats = train.groupby("_lib_n")["log_prix"].agg(["min", "median", "max"])
        prix_stats.columns = ["lp_min", "lp_med", "lp_max"]
        g = g.merge(prix_stats, left_on="_lib_n", right_index=True, how="left")

    X, feat = _build_features_fit(g, use_vendeur, use_prix and prix_stats is not None)
    y = g["target"].astype(str).values
    weights = g["weight"].values.astype(np.float32)
    # lbfgs : multinomial natif + sparse + sample_weight, sans routing de métadonnées.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42)
    clf.fit(X, y, sample_weight=weights)

    feat.update({"clf": clf, "prix_stats": prix_stats,
                 "use_vendeur": use_vendeur, "use_prix": use_prix and prix_stats is not None})
    return feat


def predire_recat(bundle: dict, df: pd.DataFrame, batch: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    df_p = pd.DataFrame({"_lib_n": df["Libelle"].map(_norm_lib)})
    if bundle["ven_cols"]:
        df_p["Vendeur_grp"] = (_vendeur_groupe(df["Vendeur"]).values
                               if "Vendeur" in df.columns else "__autre__")
    if bundle["prix_stats"] is not None:
        df_p = df_p.merge(bundle["prix_stats"], left_on="_lib_n", right_index=True, how="left")
    X = _build_features_transform(df_p, bundle)
    classes = bundle["clf"].classes_
    pred = np.empty(X.shape[0], dtype=object)
    score = np.empty(X.shape[0], dtype=float)
    for i in range(0, X.shape[0], batch):
        proba = bundle["clf"].predict_proba(X[i:i + batch])
        top = proba.argmax(axis=1)
        pred[i:i + len(top)] = classes[top]
        score[i:i + len(top)] = proba[np.arange(len(top)), top]
    return pred, score


def _commentaire(is_modif: np.ndarray, score: np.ndarray) -> list[str]:
    actions = np.where(is_modif, "Modifie", "Garde")
    tranche = np.where(score >= 0.90, "Tres probable",
              np.where(score >= 0.70, "Probable",
              np.where(score >= 0.50, "Possible", "Hesitant")))
    return [f"{a} - {t}" for a, t in zip(actions, tranche)]


def executer_recat(df: pd.DataFrame, two_pass: bool = False,
              seuil_p2: float = DEFAULT_SEUIL_P2, seuil_p1: float = DEFAULT_SEUIL_P1,
              use_vendeur: bool = True, use_prix: bool = True) -> tuple[pd.DataFrame, dict]:
    """Entraîne + prédit + applique la cascade. Retourne (df enrichi, infos).

    1 passe  : score ≥ seuil_p2 -> prédiction, sinon Nature d'origine.
    2 passes : score Pass 2 ≥ seuil_p2 -> Pass 2 ; sinon score Pass 1 ≥ seuil_p1 -> Pass 1 ;
               sinon Nature d'origine. La passe 2 est un self-training au niveau ligne :
               on ré-étiquette la cible des lignes que la passe 1 prédit autrement avec
               confiance, puis on ré-entraîne.

    Colonnes : Nature_predite, Nature_Score, Nature_Commentaire
    (+ en 2 passes : Nature_Score_Pass1, Nature_Score_Pass2).
    """
    out = df.copy()
    bundle1 = entrainer_recat(df, use_vendeur=use_vendeur, use_prix=use_prix)
    if bundle1 is None:
        return out, {"ok": False,
                     "msg": "Pas assez de Natures connues pour entraîner (≥2 classes et ≥20 lignes requises)."}

    p1_pred, p1_score = predire_recat(bundle1, df)
    nature_orig = df["Nature"].fillna("__VIDE__").astype(str).values

    if not two_pass:
        predite_raw = np.where(p1_score >= seuil_p2, p1_pred.astype(str), nature_orig)
        chosen_score = p1_score
        info_extra = {"two_pass": False, "seuil_p2": seuil_p2}
    else:
        # Self-training (niveau ligne) : les lignes dont la Nature est connue mais que la
        # passe 1 prédit autrement avec confiance sont ré-étiquetées avec la prédiction passe 1.
        p1_pred_s = pd.Series(p1_pred, index=df.index).astype(str)
        nat_known = df["Nature"].astype(str).str.strip()
        relabel = (df["Nature"].notna().values
                   & (p1_score >= seuil_p2)
                   & (p1_pred_s.values != nat_known.values))
        nature_clean = df["Nature"].astype("object").copy()
        nature_clean.loc[relabel] = p1_pred_s[relabel]
        df2 = df.copy()
        df2["Nature"] = nature_clean
        bundle2 = entrainer_recat(df2, use_vendeur=use_vendeur, use_prix=use_prix) or bundle1
        p2_pred, p2_score = predire_recat(bundle2, df)
        # Cascade : Pass 2 prioritaire, repli Pass 1, sinon Nature d'origine.
        predite_raw = np.where(
            p2_score >= seuil_p2, p2_pred.astype(str),
            np.where(p1_score >= seuil_p1, p1_pred.astype(str), nature_orig),
        )
        chosen_score = np.where(p2_score >= seuil_p2, p2_score,
                                np.where(p1_score >= seuil_p1, p1_score, p2_score))
        out["Nature_Score_Pass1"] = np.round(p1_score, 4)
        out["Nature_Score_Pass2"] = np.round(p2_score, 4)
        info_extra = {"two_pass": True, "seuil_p2": seuil_p2, "seuil_p1": seuil_p1,
                      "n_relabel": int(relabel.sum())}

    out["Nature_predite"] = pd.Series(predite_raw, index=df.index).where(
        lambda s: s != "__VIDE__", np.nan)
    out["Nature_Score"] = np.round(chosen_score, 4)
    is_modif = predite_raw != nature_orig
    out["Nature_Commentaire"] = _commentaire(is_modif, np.asarray(chosen_score))

    info = {
        "ok": True,
        "n_lignes": int(len(out)),
        "n_modifies": int(is_modif.sum()),
        "pct_modifies": float(is_modif.mean() * 100),
        "score_median": float(np.median(chosen_score)),
        "n_classes": int(len(bundle1["clf"].classes_)),
        "use_vendeur": bundle1["use_vendeur"],
        "use_prix": bundle1["use_prix"],
        **info_extra,
    }
    return out, info
