"""Calcul des statistiques descriptives du jeu de données.

Ce module regroupe les indicateurs affichés sur la page d'accueil et la page
"Données brutes". Tout est calculé dynamiquement à partir du DataFrame, et on
vérifie systématiquement la présence de chaque colonne (un fichier peut ne pas
toutes les contenir).
"""
from __future__ import annotations

import pandas as pd


def statistiques_globales(df: pd.DataFrame) -> dict:
    """Renvoie un dictionnaire d'indicateurs clés sur l'ensemble du dataset.

    Chaque indicateur n'est calculé que si la colonne nécessaire existe ;
    sinon on renvoie None (ou 0) pour rester robuste sur n'importe quel fichier.
    """
    resultats = {}
    resultats["nb_lignes"] = int(len(df))

    # Nombre de commandes distinctes (une commande = plusieurs lignes produit).
    resultats["nb_commandes_uniques"] = (
        int(df["Cod_cmd"].nunique()) if "Cod_cmd" in df.columns else None
    )
    resultats["nb_libelles_uniques"] = (
        int(df["Libelle"].nunique()) if "Libelle" in df.columns else None
    )

    # CA = chiffre d'affaires, déjà pré-calculé par le chargement (Montant x Quantité).
    if "CA" in df.columns:
        resultats["ca_total"] = float(df["CA"].sum())
    if "Quantite" in df.columns:
        resultats["qte_total"] = float(df["Quantite"].sum())

    # Période couverte (min/max de la date), seulement s'il y a des dates valides.
    if "Date" in df.columns and df["Date"].notna().any():
        resultats["periode"] = (df["Date"].min(), df["Date"].max())

    # Cardinalité des variables catégorielles.
    resultats["nb_vendeurs"] = int(df["Vendeur"].nunique()) if "Vendeur" in df.columns else None
    resultats["nb_univers"] = int(df["Univers"].nunique()) if "Univers" in df.columns else None
    resultats["nb_natures"] = int(df["Nature"].nunique()) if "Nature" in df.columns else None

    # Nombre de valeurs manquantes sur les catégories (utile pour la qualité).
    resultats["nan_univers"] = int(df["Univers"].isna().sum()) if "Univers" in df.columns else 0
    resultats["nan_nature"] = int(df["Nature"].isna().sum()) if "Nature" in df.columns else 0
    return resultats


def qualite_par_vendeur(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de qualité des données par vendeur : volume, manquants, CA.

    Permet de repérer les vendeurs qui laissent beaucoup d'Univers/Nature vides.
    Renvoie un DataFrame vide si la colonne Vendeur n'existe pas.
    """
    if "Vendeur" not in df.columns:
        return pd.DataFrame()

    # On agrège ligne par vendeur : nb de lignes, nb de manquants, CA total.
    agregat = df.groupby("Vendeur").agg(
        nb_lignes=("Libelle", "size"),
        nan_univers=("Univers", lambda s: s.isna().sum()),
        nan_nature=("Nature", lambda s: s.isna().sum()),
        ca=("CA", "sum") if "CA" in df.columns else ("Libelle", "size"),
    )

    # On convertit les manquants en pourcentage pour comparer les vendeurs entre eux.
    agregat["pct_nan_univers"] = (agregat["nan_univers"] / agregat["nb_lignes"] * 100).round(1)
    agregat["pct_nan_nature"] = (agregat["nan_nature"] / agregat["nb_lignes"] * 100).round(1)
    return agregat.sort_values("nb_lignes", ascending=False)
