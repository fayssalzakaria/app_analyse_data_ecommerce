"""Recatégorisation générique d'une colonne catégorielle (Nature ou Univers).

Le modèle prédit une cible (`Nature` ou `Univers`) à partir du libellé produit :
- TF-IDF mots (1-2 grammes) + TF-IDF caractères (char_wb 3-5),
- + variables catégorielles : vendeur (regroupé) et éventuelles colonnes
  auxiliaires (par ex. la Nature corrigée quand on prédit l'Univers),
- + statistiques de prix par libellé (si dispo),
- régression logistique entraînée sur les lignes dont la cible est connue
  (agrégées avec un poids = nombre d'occurrences).

Aucune valeur n'est codée en dur : tout est appris à partir du dataset fourni.
Sortie : <cible>_predite, <cible>_Score, <cible>_Commentaire.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Seuils de la cascade. En 1 passe, seul DEFAULT_SEUIL_P2 sert.
# En 2 passes : score Pass 2 >= seuil_p2 -> Pass 2 ; sinon score Pass 1 >= seuil_p1 -> Pass 1 ;
# sinon on conserve la valeur d'origine.
DEFAULT_SEUIL_P2 = 0.50
DEFAULT_SEUIL_P1 = 0.80
DEFAULT_SEUIL = DEFAULT_SEUIL_P2  # rétro-compat
VENDEUR_TOP_K = 8
RECAT_VERSION = "recat-generic-v3"


def colonne_predite(cible: str) -> str:
    """Nom de la colonne de prédiction, accordé en genre.

    « Univers » est masculin -> 'Univers_predit' ; « Nature » est féminin -> 'Nature_predite'.
    """
    return "Univers_predit" if cible.lower() == "univers" else f"{cible}_predite"


def _norm_lib(s) -> str | None:
    """Normalise un libellé (minuscules, accents conservés, espaces compactés)."""
    if pd.isna(s):
        return None
    return " ".join(unicodedata.normalize("NFC", str(s).strip().lower()).split())


def _vendeur_groupe(vendeur: pd.Series, top_k: int = VENDEUR_TOP_K) -> pd.Series:
    """Garde les top_k vendeurs les plus fréquents, regroupe le reste sous '__autre__'."""
    v = vendeur.fillna("__na__").astype(str)
    top = v.value_counts().head(top_k).index
    return v.where(v.isin(top), "__autre__")


def _log_prix(df: pd.DataFrame) -> pd.Series | None:
    """Log du prix unitaire (Montant / Quantité) si les colonnes existent."""
    if {"Montant_cmd", "Quantite"}.issubset(df.columns):
        qte = pd.to_numeric(df["Quantite"], errors="coerce").replace(0, 1)
        prix_unit = pd.to_numeric(df["Montant_cmd"], errors="coerce") / qte
        return np.log1p(prix_unit.fillna(0).clip(lower=0))
    if "Montant_cmd" in df.columns:
        return np.log1p(pd.to_numeric(df["Montant_cmd"], errors="coerce").fillna(0).clip(lower=0))
    return None


def _features_categorielles(df: pd.DataFrame, use_vendeur: bool,
                            colonnes_aux: list[str] | None) -> dict[str, pd.Series]:
    """Construit le dictionnaire {nom_feature: valeurs} des variables catégorielles.

    Le vendeur est regroupé (top-K) ; les colonnes auxiliaires (ex: Nature pour
    prédire l'Univers) sont prises telles quelles.
    """
    feats: dict[str, pd.Series] = {}
    if use_vendeur and "Vendeur" in df.columns:
        feats["vendeur"] = _vendeur_groupe(df["Vendeur"]).reset_index(drop=True)
    for col in (colonnes_aux or []):
        if col in df.columns:
            feats[col] = df[col].fillna("__na__").astype(str).reset_index(drop=True)
    return feats


def entrainer_recat(df: pd.DataFrame, cible: str = "Nature", use_vendeur: bool = True,
                    use_prix: bool = True, colonnes_aux: list[str] | None = None) -> dict | None:
    """Entraîne le modèle pour prédire `cible`. None si pas assez de données.

    colonnes_aux : variables catégorielles supplémentaires en entrée
    (ex: ["Nature_predite"] pour prédire l'Univers à partir de la Nature corrigée).
    """
    if "Libelle" not in df.columns or cible not in df.columns:
        return None

    work = pd.DataFrame({"_lib_n": df["Libelle"].map(_norm_lib).reset_index(drop=True),
                         "target": df[cible].reset_index(drop=True)})
    cat_feats = _features_categorielles(df, use_vendeur, colonnes_aux)
    for nom, valeurs in cat_feats.items():
        work[f"cat__{nom}"] = valeurs.values
    lp = _log_prix(df) if use_prix else None
    if lp is not None:
        work["log_prix"] = lp.reset_index(drop=True).values

    train = work[work["_lib_n"].notna() & work["target"].notna()].copy()
    if train["target"].nunique() < 2 or len(train) < 20:
        return None  # pas assez de signal pour entraîner

    cat_cols_noms = [c for c in train.columns if c.startswith("cat__")]
    g = train.groupby(["_lib_n", "target"] + cat_cols_noms, dropna=False).size().reset_index(name="weight")

    prix_stats = None
    if "log_prix" in train.columns:
        prix_stats = train.groupby("_lib_n")["log_prix"].agg(["min", "median", "max"])
        prix_stats.columns = ["lp_min", "lp_med", "lp_max"]
        g = g.merge(prix_stats, left_on="_lib_n", right_index=True, how="left")

    # --- Construction des features (texte + catégorielles + prix) ---
    vect_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5,
                             max_features=15000, sublinear_tf=True)
    Xw = vect_w.fit_transform(g["_lib_n"])
    vect_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.5,
                             max_features=15000, sublinear_tf=True)
    Xc = vect_c.fit_transform(g["_lib_n"])
    blocs = [Xw, Xc]

    cat_dummies_cols: dict[str, list[str]] = {}
    for col in cat_cols_noms:
        d = pd.get_dummies(g[col], prefix=col).astype(np.float32)
        cat_dummies_cols[col] = d.columns.tolist()
        blocs.append(csr_matrix(d.values))

    prix_mean = prix_std = None
    if prix_stats is not None:
        prix_arr = g[["lp_min", "lp_med", "lp_max"]].fillna(0).values.astype(np.float32)
        prix_mean = prix_arr.mean(axis=0)
        prix_std = prix_arr.std(axis=0).clip(min=1e-6)
        blocs.append(csr_matrix((prix_arr - prix_mean) / prix_std))

    X = hstack(blocs).tocsr()
    X.sort_indices(); X.sum_duplicates()

    y = g["target"].astype(str).values
    poids = g["weight"].values.astype(np.float32)
    # lbfgs : multinomial natif + sparse + sample_weight, sans routing de métadonnées.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42)
    clf.fit(X, y, sample_weight=poids)

    return {
        "cible": cible, "use_vendeur": use_vendeur, "colonnes_aux": colonnes_aux or [],
        "vect_w": vect_w, "vect_c": vect_c,
        "cat_dummies_cols": cat_dummies_cols, "prix_stats": prix_stats,
        "prix_mean": prix_mean, "prix_std": prix_std, "clf": clf,
    }


def predire_recat(bundle: dict, df: pd.DataFrame, batch: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    """Prédit la cible (label + score de confiance) pour chaque ligne du df."""
    df_p = pd.DataFrame({"_lib_n": df["Libelle"].map(_norm_lib).reset_index(drop=True)})
    cat_feats = _features_categorielles(df, bundle["use_vendeur"], bundle["colonnes_aux"])
    for nom, valeurs in cat_feats.items():
        df_p[f"cat__{nom}"] = valeurs.values
    if bundle["prix_stats"] is not None:
        df_p = df_p.merge(bundle["prix_stats"], left_on="_lib_n", right_index=True, how="left")

    Xw = bundle["vect_w"].transform(df_p["_lib_n"].fillna(""))
    Xc = bundle["vect_c"].transform(df_p["_lib_n"].fillna(""))
    blocs = [Xw, Xc]
    for col, cols in bundle["cat_dummies_cols"].items():
        serie = df_p[col] if col in df_p.columns else pd.Series(["__na__"] * len(df_p))
        d = pd.get_dummies(serie, prefix=col).reindex(columns=cols, fill_value=0).astype(np.float32)
        blocs.append(csr_matrix(d.values))
    if bundle["prix_mean"] is not None:
        prix_arr = df_p[["lp_min", "lp_med", "lp_max"]].fillna(0).values.astype(np.float32)
        blocs.append(csr_matrix((prix_arr - bundle["prix_mean"]) / bundle["prix_std"]))
    X = hstack(blocs).tocsr()

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
    """Commentaire en 7 tranches : (Garde/Modifie) x (niveau de confiance)."""
    actions = np.where(is_modif, "Modifie", "Garde")
    tranche = np.where(score >= 0.90, "Tres probable",
              np.where(score >= 0.70, "Probable",
              np.where(score >= 0.50, "Possible", "Hesitant")))
    return [f"{a} - {t}" for a, t in zip(actions, tranche)]


def executer_recat(df: pd.DataFrame, cible: str = "Nature", two_pass: bool = False,
                   seuil_p2: float = DEFAULT_SEUIL_P2, seuil_p1: float = DEFAULT_SEUIL_P1,
                   use_vendeur: bool = True, use_prix: bool = True,
                   colonnes_aux: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Entraîne + prédit + applique la cascade. Retourne (df enrichi, infos).

    Colonnes ajoutées : <cible>_predite, <cible>_Score, <cible>_Commentaire
    (+ en 2 passes : <cible>_Score_Pass1, <cible>_Score_Pass2).
    """
    out = df.copy()
    col_pred = colonne_predite(cible)
    col_score = f"{cible}_Score"
    col_comm = f"{cible}_Commentaire"

    bundle1 = entrainer_recat(df, cible=cible, use_vendeur=use_vendeur,
                              use_prix=use_prix, colonnes_aux=colonnes_aux)
    if bundle1 is None:
        return out, {"ok": False, "cible": cible,
                     "msg": f"Pas assez de valeurs connues de '{cible}' pour entraîner "
                            "(>=2 classes et >=20 lignes requises)."}

    p1_pred, p1_score = predire_recat(bundle1, df)
    valeur_orig = df[cible].fillna("__VIDE__").astype(str).values

    if not two_pass:
        predite_raw = np.where(p1_score >= seuil_p2, p1_pred.astype(str), valeur_orig)
        chosen_score = p1_score
        info_extra = {"two_pass": False, "seuil_p2": seuil_p2}
    else:
        # Self-training (niveau ligne) : on ré-étiquette les lignes que la passe 1
        # prédit autrement avec confiance, puis on ré-entraîne.
        p1_pred_s = pd.Series(p1_pred, index=df.index).astype(str)
        orig_known = df[cible].astype(str).str.strip()
        relabel = (df[cible].notna().values
                   & (p1_score >= seuil_p2)
                   & (p1_pred_s.values != orig_known.values))
        cible_clean = df[cible].astype("object").copy()
        cible_clean.loc[relabel] = p1_pred_s[relabel]
        df2 = df.copy()
        df2[cible] = cible_clean
        bundle2 = entrainer_recat(df2, cible=cible, use_vendeur=use_vendeur,
                                  use_prix=use_prix, colonnes_aux=colonnes_aux) or bundle1
        p2_pred, p2_score = predire_recat(bundle2, df)
        predite_raw = np.where(
            p2_score >= seuil_p2, p2_pred.astype(str),
            np.where(p1_score >= seuil_p1, p1_pred.astype(str), valeur_orig),
        )
        chosen_score = np.where(p2_score >= seuil_p2, p2_score,
                                np.where(p1_score >= seuil_p1, p1_score, p2_score))
        out[f"{cible}_Score_Pass1"] = np.round(p1_score, 4)
        out[f"{cible}_Score_Pass2"] = np.round(p2_score, 4)
        info_extra = {"two_pass": True, "seuil_p2": seuil_p2, "seuil_p1": seuil_p1,
                      "n_relabel": int(relabel.sum())}

    out[col_pred] = pd.Series(predite_raw, index=df.index).where(lambda s: s != "__VIDE__", np.nan)
    out[col_score] = np.round(chosen_score, 4)
    is_modif = predite_raw != valeur_orig
    out[col_comm] = _commentaire(is_modif, np.asarray(chosen_score))

    info = {
        "ok": True, "cible": cible,
        "n_lignes": int(len(out)),
        "n_modifies": int(is_modif.sum()),
        "pct_modifies": float(is_modif.mean() * 100),
        "score_median": float(np.median(chosen_score)),
        "n_classes": int(len(bundle1["clf"].classes_)),
        "use_vendeur": use_vendeur, "use_prix": use_prix,
        **info_extra,
    }
    return out, info
