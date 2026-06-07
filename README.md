---
title: Analyse Dataset E-commerce
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Analyse et purification de dataset e-commerce

Application Streamlit multi-pages pour explorer un jeu de données e-commerce, fiabiliser sa catégorisation (Nature) et en extraire des attributs (couleur, dimension).

Elle fonctionne avec n'importe quel fichier tabulaire (Excel, CSV, Parquet) comportant les colonnes de base décrites plus bas. Un petit jeu de démonstration synthétique est inclus pour la prise en main immédiate.

**Démo en ligne (Hugging Face Spaces)** : https://huggingface.co/spaces/Fayzk92/analyse-dataset-ecommerce

## Fonctionnalités

- Statistiques descriptives : KPI, évolution temporelle, répartition par vendeur / univers / nature, qualité des données.
- Recatégorisation de la `Nature` entraînée sur le fichier chargé (TF-IDF + régression logistique), en une ou deux passes.
- Recatégorisation de l'`Univers`, déduite de la Nature corrigée pour rester cohérente avec la hiérarchie (Univers ⊃ Nature).
- Extraction de couleur et de dimension à partir des libellés produits.
- Recherche multi-critères (texte libre + prix / couleur / dimension / vendeur).
- Visualisations : graphe relationnel, projection 2D/3D, diagramme de Sankey.
- Export du dataset enrichi en CSV, Excel ou Parquet.

## Installation

Prérequis : Python 3.9 ou plus.

```bash
cd app_analyse_data_ecommerce

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501. Aucun fichier n'est chargé automatiquement : déposez le vôtre dans la barre latérale, ou cliquez sur « Charger le jeu de démo » pour tester avec le jeu synthétique fourni (`data/sample/sample_ecommerce.csv`).

## Données attendues

Les noms de colonnes sont normalisés automatiquement (accents, casse, alias courants). Schéma de référence :

| Colonne | Rôle |
|---|---|
| `Cod_cmd` | Identifiant de commande |
| `Libelle` | Libellé produit (texte analysé) |
| `Vendeur` | Vendeur / marchand |
| `Univers` | Catégorie de premier niveau |
| `Nature` | Catégorie de second niveau (cible de fiabilisation) |
| `Date_cmd` | Date de commande (date ou numéro de série Excel) |
| `Montant_cmd` | Montant |
| `Quantite` | Quantité |
| `Prix_transport` | Frais de transport |
| `Delai_transport` | Délai annoncé |

Les colonnes absentes sont gérées sans erreur : chaque page n'affiche que ce qu'elle peut calculer. Formats lus : `.xlsb`, `.xlsx`, `.xls`, `.csv` (séparateur auto-détecté), `.parquet`.

### Confidentialité

Le `.gitignore` exclut les classeurs Excel (`*.xlsb`, `*.xls`, `*.xlsx`), `data/livrables/*.parquet` et tous les caches. Ne versionnez pas de données réelles sur un dépôt public. Seul le jeu synthétique `data/sample/sample_ecommerce.csv` est suivi ; il ne contient aucune donnée réelle.

## Pages

| # | Page | Description |
|---|------|-------------|
| Accueil | `app.py` | Sélection du fichier source et aperçu |
| 0 | Recherche sémantique | Recherche multi-critères + voisins TF-IDF, sur données brutes ou recatégorisées |
| 1 | Données brutes | KPI, séries temporelles, répartitions, qualité par vendeur |
| 2 | Visualisation graphe | Graphe relationnel, UMAP, Sankey, avant/après |
| 3 | Catégorisation Nature / Univers | Recatégorisation (bouton) : Nature (1 ou 2 passes) et Univers (cohérent avec la Nature) |
| 4 | Effet de la correction | V de Cramér, matrices, taux de correction par vendeur |
| 5 | Extraction couleur | Couleurs extraites des libellés (bouton) |
| 6 | Extraction dimension | Dimensions L x l x H, diamètre, cote isolée (bouton) |
| 7 | Export | Téléchargement du dataset enrichi (CSV / Excel / Parquet) |

### Enrichissement à la demande

L'application calcule l'enrichissement sur le fichier chargé, via trois actions indépendantes :

1. Recatégorisation Nature (page 3). Un modèle TF-IDF (mots et caractères), complété par le vendeur et le prix lorsqu'ils sont présents, est entraîné sur les lignes dont la `Nature` est connue. Sortie : `Nature_predite`, `Nature_Score`, `Nature_Commentaire`.
   - Une passe : si le score est supérieur ou égal au seuil, on applique la prédiction, sinon on conserve la Nature d'origine.
   - Deux passes (option) : self-training. Les lignes que la passe 1 prédit autrement avec confiance sont ré-étiquetées, puis le modèle est ré-entraîné. Cascade : score Pass 2 supérieur ou égal au seuil Pass 2 (0,50 par défaut), sinon score Pass 1 supérieur ou égal au seuil Pass 1 (0,80 par défaut), sinon Nature d'origine. Seuils réglables ; colonnes `Nature_Score_Pass1` et `Nature_Score_Pass2` ajoutées.
1bis. Recatégorisation Univers (page 3, après la Nature). On apprend la table `Nature → Univers majoritaire` et on attribue à chaque ligne l'Univers de sa Nature corrigée (cohérence garantie : une Nature donnée ne peut pointer que vers un seul Univers). Pour les lignes sans Nature exploitable, un modèle libellé→Univers prend le relais. Sortie : `Univers_predite`, `Univers_Score`, `Univers_Commentaire`.
2. Extraction couleur (page 5) : `couleur_extraite`, `Couleur_Commentaire`, `nb_couleurs_detectees`, etc.
3. Extraction dimension (page 6) : `dim_label`, `Dimension_Commentaire`, diamètre, cote isolée, etc.

Chaque action ajoute ses colonnes au dataset de travail conservé en session. Au changement de fichier source, tout l'état dérivé (enrichissements, filtres, export, caches) est réinitialisé. La page 4 nécessite la recatégorisation ; ses sections couleur et dimension s'activent une fois ces calculs effectués. La page 0 permet de comparer la recherche sur les données brutes et sur les Natures recatégorisées (la source « corrigé » nécessite d'avoir lancé la recatégorisation).

Un fichier `data/livrables/livrable_final.parquet` déjà enrichi reste utilisable : il suffit de le charger comme fichier source.

## Architecture

```
app_analyse_data_ecommerce/
├── app.py                  Accueil et sélection du fichier
├── requirements.txt
├── core/                   Logique métier
│   ├── data_loader.py      Chargement, normalisation, reset au changement de fichier
│   ├── recat.py            Recatégorisation générique (Nature ou Univers, 1 ou 2 passes)
│   ├── enrich.py           Orchestration des calculs et dataset de travail
│   ├── extract.py          Extraction couleur et dimension
│   ├── metrics.py          Statistiques globales
│   ├── search.py           Recherche TF-IDF et suggestion
│   ├── embeddings.py       Encodage sémantique et réduction de dimension
│   └── cache_utils.py      Cache disque
├── pages/                  Huit pages Streamlit (0 à 7)
└── data/
    ├── sample/             Jeu de démonstration synthétique (suivi)
    ├── cache/              Caches (ignoré par git)
    └── livrables/          Dataset enrichi éventuel (ignoré par git)
```

## Dépendances optionnelles

Par défaut, toute l'analyse repose sur TF-IDF (scikit-learn) ; `requirements.txt` n'installe rien de lourd. Les extras ci-dessous ne sont pas nécessaires :

- Projection sémantique UMAP (page 2) : `sentence-transformers` et `umap-learn`, non installés par défaut. Sans eux, l'app utilise TF-IDF + SVD. Pour les activer :
  ```bash
  pip install sentence-transformers umap-learn
  export APP_USE_SENTENCE_TRANSFORMERS=1
  export APP_USE_UMAP=1
  ```

## Adapter à un autre jeu de données

- `core/data_loader.py` : ajouter des alias si vos en-têtes diffèrent du schéma de référence.
- `core/extract.py` : étendre les palettes de couleurs ou les règles de dimension.
- Les seuils et sélections par défaut des pages 3 et 4 sont calculés à partir du fichier chargé.

## Déploiement (Hugging Face Spaces)

L'application est hébergée sur **Hugging Face Spaces** : https://huggingface.co/spaces/Fayzk92/analyse-dataset-ecommerce

Le Space utilise le **SDK Streamlit** : c'est l'en-tête YAML en haut de ce README (`sdk: streamlit`, `app_file: app.py`) qui le configure, et `requirements.txt` qui fournit les dépendances. Aucune configuration supplémentaire n'est nécessaire.

**Mise à jour automatique** : un workflow GitHub Actions (`.github/workflows/sync-to-huggingface.yml`) pousse le dépôt vers le Space à chaque `git push` sur `main` ou `master`. Il faut pour cela :

1. créer un jeton Hugging Face (Settings → Access Tokens, droits **Write**) ;
2. l'ajouter en secret du dépôt GitHub sous le nom `HF_TOKEN` (Settings → Secrets and variables → Actions).

À chaque push GitHub, le Space se reconstruit et redéploie tout seul.

> Sur le tier gratuit, prévoir une mémoire limitée (~1 Go sur Streamlit Cloud, ~16 Go sur Hugging Face CPU). Pour de très gros fichiers, préférer un échantillon ou un hébergement avec plus de RAM.

## Licence

MIT. Voir le fichier [LICENSE](LICENSE).
