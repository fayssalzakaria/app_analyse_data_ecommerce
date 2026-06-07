"""Extraction couleur + dimension depuis les libellés produit."""
from __future__ import annotations
import hashlib
import os
import re
import subprocess
import unicodedata
import pandas as pd


PALETTE_BASE = [
    "blanc", "noir", "gris", "beige", "marron", "brun", "rouge", "bleu",
    "vert", "jaune", "orange", "rose", "violet", "or", "argent", "cuivre",
    "chene", "wenge", "naturel", "taupe", "ivoire", "creme", "ecru",
    "anthracite", "bordeaux", "kaki", "turquoise",
]

PALETTE_FINE = PALETTE_BASE + [
    "blanc mat", "blanc casse", "blanc laque", "blanc brillant",
    "noir mat", "noir laque", "noir brillant",
    "gris anthracite", "gris clair", "gris fonce", "gris souris", "gris perle",
    "chene naturel", "chene clair", "chene fonce", "chene massif", "chene blanchi",
    "marron clair", "marron fonce",
    "bleu marine", "bleu nuit", "bleu ciel", "bleu canard", "bleu petrole",
    "vert sapin", "vert anis", "vert d'eau", "vert olive",
    "rose pale", "rose poudre", "rose fuchsia",
    "rouge bordeaux", "rouge cerise",
]

PALETTE_EXTRA = [
    "terracotta", "camel", "sable", "lin", "noyer", "teck", "acacia", "bois clair",
]

COLOR_ALIASES = {
    "ch ne": "chene",
    "ch ne clair": "chene clair",
    "ch ne fonce": "chene fonce",
    "ch ne massif": "chene massif",
    "ch ne blanchi": "chene blanchi",
    "blanche": "blanc",
    "blanches": "blanc",
    "blancs": "blanc",
    "noire": "noir",
    "noires": "noir",
    "noirs": "noir",
    "grise": "gris",
    "grises": "gris",
    "bleue": "bleu",
    "bleues": "bleu",
    "bleus": "bleu",
    "verte": "vert",
    "vertes": "vert",
    "verts": "vert",
    "beiges": "beige",
    "taupes": "taupe",
    "argente": "argent",
    "argentee": "argent",
    "argentees": "argent",
    "argentes": "argent",
    "blan": "blanc",
    "noi": "noir",
    "girs": "gris",
    "grsi": "gris",
    "antracite": "anthracite",
    "anthracit": "anthracite",
    "terracota": "terracotta",
}

AMBIGUOUS_FINISHES = {"chene", "noyer", "teck", "acacia", "bois clair", "lin", "sable"}
COLOR_ALGO_VERSION = "colors-2026-05-20-v1"
MATERIAL_CONTEXT_RE = re.compile(
    r"\b(tissu|toile|effet|imitation|faux|matiere|rev[eê]tement|housse|rideau|voilage|coton|polyester)\b"
)
COLOR_CONTEXT_RE = re.compile(r"\b(couleur|coloris|teinte|teinte|ton|tons|uni|chine|chinee)\b")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _strip_accents(s.lower())


def construire_regex_couleur(palette: list[str]) -> re.Pattern:
    """Compile une regex qui matche n'importe quelle couleur de la palette, longue d'abord."""
    sorted_palette = sorted(set(palette), key=lambda x: -len(x))
    escaped = [re.escape(p) for p in sorted_palette]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def extraire_couleur(libelle: str, regex: re.Pattern) -> str | None:
    """Retourne la première couleur trouvée (ou None)."""
    s = _normalize(libelle)
    m = regex.search(s)
    return m.group(1) if m else None


def extraire_couleurs_serie(libelles: pd.Series, fine: bool = False) -> pd.Series:
    """Vectorisé."""
    palette = PALETTE_FINE if fine else PALETTE_BASE
    regex = construire_regex_couleur(palette)
    return libelles.fillna("").astype(str).apply(lambda s: extraire_couleur(s, regex))


def _enhanced_color_lookup(fine: bool = True) -> tuple[re.Pattern, dict[str, str], set[str]]:
    palette = (PALETTE_FINE if fine else PALETTE_BASE) + PALETTE_EXTRA
    canonical = {_normalize(c): _normalize(c) for c in palette}
    aliases = {_normalize(k): _normalize(v) for k, v in COLOR_ALIASES.items()}
    lookup = {**canonical, **aliases}

    variants = sorted(lookup, key=lambda x: (-len(x), x))
    regex = re.compile(r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b", flags=re.IGNORECASE)
    return regex, lookup, set(canonical)


def extraire_couleur_details(libelle: str, fine: bool = True) -> dict:
    """Extraction couleur enrichie et explicable.

    La normalisation conserve le traitement actuel des accents, puis ajoute des
    alias sûrs, les formes féminines/plurielles et la détection multi-couleur.
    """
    regex, lookup, canonical = _enhanced_color_lookup(fine=fine)
    s = _normalize(libelle)
    colors = []
    raw_matches = []
    sources = []

    for match in regex.finditer(s):
        raw = match.group(1)
        color = lookup.get(raw, raw)
        raw_matches.append(raw)
        colors.append(color)
        sources.append("exact" if raw in canonical and raw == color else "alias")

    unique_colors = []
    for color in colors:
        if color not in unique_colors:
            unique_colors.append(color)

    if not unique_colors:
        return {
            "couleur_principale": None,
            "couleurs_detectees": None,
            "nb_couleurs_detectees": 0,
            "couleur_statut": "non_detecte",
            "couleur_source": None,
            "couleur_matchs_raw": None,
        }

    primary = unique_colors[0]
    source = "alias_sur" if "alias" in sources else "exact"
    is_ambiguous_finish = primary in AMBIGUOUS_FINISHES or primary.startswith("chene ")
    if len(unique_colors) > 1:
        status = "multi_couleur"
    elif is_ambiguous_finish:
        status = "ambigu_finition"
    else:
        status = source

    return {
        "couleur_principale": primary,
        "couleurs_detectees": ", ".join(unique_colors),
        "nb_couleurs_detectees": len(unique_colors),
        "couleur_statut": status,
        "couleur_source": source,
        "couleur_matchs_raw": ", ".join(raw_matches),
    }


def extraire_couleur_details_serie(libelles: pd.Series, fine: bool = True) -> pd.DataFrame:
    """Vectorisé : retourne les colonnes enrichies de détection couleur."""
    cols = [
        "couleur_principale",
        "couleurs_detectees",
        "nb_couleurs_detectees",
        "couleur_statut",
        "couleur_source",
        "couleur_matchs_raw",
    ]
    rows = libelles.fillna("").astype(str).apply(lambda s: extraire_couleur_details(s, fine=fine))
    return pd.DataFrame(list(rows), index=libelles.index).reindex(columns=cols)


def _window_around(text: str, needle: str, size: int = 35) -> str:
    pos = text.find(needle)
    if pos < 0:
        return ""
    return text[max(0, pos - size): min(len(text), pos + len(needle) + size)]


def extraire_couleur_contextuelle(libelle: str, fine: bool = True) -> dict:
    """Niveau 3 : decision contextuelle prudente au-dessus du socle renforce.

    Le but n'est pas de deviner partout, mais de separer couleur probable,
    matiere probable et finition bois quand les indices de contexte sont forts.
    """
    details = extraire_couleur_details(libelle, fine=fine)
    s = _normalize(libelle)
    detected = [c.strip() for c in (details["couleurs_detectees"] or "").split(",") if c.strip()]
    raw = [c.strip() for c in (details["couleur_matchs_raw"] or "").split(",") if c.strip()]

    if not detected:
        return {
            **details,
            "couleur_niveau3": None,
            "finition_niveau3": None,
            "decision_niveau3": "non_detecte",
            "score_niveau3": 0.0,
            "raison_niveau3": "aucun terme couleur trouve",
        }

    safe_colors = [c for c in detected if c not in AMBIGUOUS_FINISHES and not c.startswith("chene ")]
    ambiguous = [c for c in detected if c in AMBIGUOUS_FINISHES or c.startswith("chene ")]

    # Si une couleur non ambigue existe avec une finition/matiere, elle gagne.
    if safe_colors and ambiguous:
        return {
            **details,
            "couleur_niveau3": safe_colors[0],
            "finition_niveau3": ", ".join(ambiguous),
            "decision_niveau3": "couleur_avec_finition",
            "score_niveau3": 0.85,
            "raison_niveau3": "couleur non ambigue presente avec finition/matiere",
        }

    primary = detected[0]
    primary_raw = raw[0] if raw else primary
    context = _window_around(s, primary_raw)
    has_material_context = bool(MATERIAL_CONTEXT_RE.search(context))
    has_color_context = bool(COLOR_CONTEXT_RE.search(context))

    if ambiguous:
        if has_color_context and not has_material_context:
            return {
                **details,
                "couleur_niveau3": primary,
                "finition_niveau3": None,
                "decision_niveau3": "couleur_contextuelle",
                "score_niveau3": 0.75,
                "raison_niveau3": "mot couleur/coloris/teinte proche du terme ambigu",
            }
        if has_material_context:
            return {
                **details,
                "couleur_niveau3": None,
                "finition_niveau3": ", ".join(ambiguous),
                "decision_niveau3": "matiere_probable",
                "score_niveau3": 0.80,
                "raison_niveau3": "mot tissu/effet/imitation/faux proche du terme ambigu",
            }
        return {
            **details,
            "couleur_niveau3": None,
            "finition_niveau3": ", ".join(ambiguous),
            "decision_niveau3": "ambigu_a_revoir",
            "score_niveau3": 0.45,
            "raison_niveau3": "terme ambigu sans contexte suffisant",
        }

    return {
        **details,
        "couleur_niveau3": primary,
        "finition_niveau3": None,
        "decision_niveau3": details["couleur_statut"],
        "score_niveau3": 0.90 if details["couleur_statut"] in {"exact", "alias_sur"} else 0.80,
        "raison_niveau3": "couleur non ambigue",
    }


def extraire_couleur_contextuelle_serie(libelles: pd.Series, fine: bool = True) -> pd.DataFrame:
    cols = [
        "couleur_principale",
        "couleurs_detectees",
        "nb_couleurs_detectees",
        "couleur_statut",
        "couleur_source",
        "couleur_matchs_raw",
        "couleur_niveau3",
        "finition_niveau3",
        "decision_niveau3",
        "score_niveau3",
        "raison_niveau3",
    ]
    rows = libelles.fillna("").astype(str).apply(lambda s: extraire_couleur_contextuelle(s, fine=fine))
    return pd.DataFrame(list(rows), index=libelles.index).reindex(columns=cols)


def _cache_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return "no-file-hash"


def _git_commit_hash() -> str:
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip() or "no-git-head"
    except Exception:
        return "no-git-head"


def cle_cache_couleur(dataset_path: str | None, fine: bool, level: str) -> str:
    """Stable cache key invalidated by dataset metadata, palette and algo version."""
    if dataset_path and os.path.exists(dataset_path):
        stat = os.stat(dataset_path)
        raw = f"{os.path.abspath(dataset_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    else:
        raw = "no-dataset-path"
    code_hash = _file_hash(__file__)
    git_hash = _git_commit_hash()
    raw += (
        f"|fine={int(fine)}|level={level}|version={COLOR_ALGO_VERSION}"
        f"|extract_py={code_hash}|git={git_hash}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def chemin_cache_couleur(dataset_path: str | None, fine: bool, level: str) -> str:
    return os.path.join(_cache_dir(), f"colors_{level}_{cle_cache_couleur(dataset_path, fine, level)}.pkl")


def charger_cache_couleur(dataset_path: str | None, fine: bool, level: str) -> pd.DataFrame | None:
    path = chemin_cache_couleur(dataset_path, fine, level)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def sauvegarder_cache_couleur(dataset_path: str | None, fine: bool, level: str, data: pd.DataFrame) -> None:
    path = chemin_cache_couleur(dataset_path, fine, level)
    try:
        data.to_pickle(path)
    except Exception:
        pass


def construire_niveaux_couleur_12(libelles: pd.Series, counts: pd.Series, fine: bool = True) -> pd.DataFrame:
    libelles = libelles.fillna("").astype(str).reset_index(drop=True)
    out = pd.DataFrame({"Libelle": libelles, "nb_lignes": counts.reset_index(drop=True)})
    out["niveau1_avant"] = extraire_couleurs_serie(libelles, fine=fine).values
    details = extraire_couleur_details_serie(libelles, fine=fine).reset_index(drop=True)
    return pd.concat([out, details], axis=1)


def ajouter_niveau_couleur_3(levels12: pd.DataFrame, fine: bool = True) -> pd.DataFrame:
    context = extraire_couleur_contextuelle_serie(levels12["Libelle"], fine=fine).reset_index(drop=True)
    cols = ["couleur_niveau3", "finition_niveau3", "decision_niveau3", "score_niveau3", "raison_niveau3"]
    out = levels12.copy()
    for col in cols:
        out[col] = context[col]
    return out


# ---------------- Dimensions ----------------

# Capture AxB ou AxBxC, séparateurs : x, X, *, ×, avec ou sans espaces, décimales OK (50.5, 50,5)
_DIM_RE = re.compile(
    r"(?<!\d)(\d{1,4}(?:[.,]\d{1,2})?)\s*[xX×\*]\s*(\d{1,4}(?:[.,]\d{1,2})?)"
    r"(?:\s*[xX×\*]\s*(\d{1,4}(?:[.,]\d{1,2})?))?(?!\d)"
)
_UNIT_RE = re.compile(r"\b(cm|mm|m|inch|pouces?|po)\b", re.IGNORECASE)
_DIM_DECIMAL_SPACED_RE = re.compile(
    r"(?P<raw>"
    r"\d{1,4}(?:\s+\d{1,2})?\s*[xXÃ—\*]\s*"
    r"\d{1,4}(?:\s+\d{1,2})?"
    r"(?:\s*[xXÃ—\*]\s*\d{1,4}(?:\s+\d{1,2})?)?"
    r"\s*(?:cm|mm|m\b)?"
    r")",
    re.IGNORECASE,
)
_POWER_DIM_RE = re.compile(
    r"\b\d{1,4}(?:[.,]\d{1,2})?\s*[xXÃ—\*]\s*\d{1,5}(?:[.,]\d{1,2})?\s*(?:w|kw|v|hz|mah|ah)\b",
    re.IGNORECASE,
)
_RESOLUTION_WORDS_RE = re.compile(r"\b(resolution|pixels?|full\s*hd|uhd|4k|8k|hz|ecran|moniteur)\b", re.IGNORECASE)
_ELECTRIC_UNIT_AFTER_RE = re.compile(r"^\s*(?:w|kw|v|hz|mah|ah)\b", re.IGNORECASE)
_UNIT_AFTER_RE = re.compile(r"^\s*(cm|mm|m|inch|pouces?|po)\b", re.IGNORECASE)
_DIAM_H_RE = re.compile(
    r"\b(?:diam(?:etre)?|diam\.?|[Ã¸Ã˜])\s*"
    r"(?P<D>\d{1,4}(?:[.,]\d{1,2})?(?:\s+\d)?)\s*(?P<unit_d>cm|mm|m)?"
    r"\s*[xXÃ—\*]?\s*h(?:auteur)?\.?\s*"
    r"(?P<H>\d{1,4}(?:[.,]\d{1,2})?(?:\s+\d)?)\s*(?P<unit_h>cm|mm|m)?\b",
    re.IGNORECASE,
)
_DIAM_RE = re.compile(
    r"\b(?:diam(?:etre)?|diam\.?|[Ã¸Ã˜])\s*"
    r"(?P<D>\d{1,4}(?:[.,]\d{1,2})?(?:\s+\d)?)\s*(?P<unit>cm|mm|m)?\b",
    re.IGNORECASE,
)
_SIMPLE_DIM_RE = re.compile(r"\b(?P<value>\d{1,4}(?:[.,]\d{1,2})?)\s*(?P<unit>cm|mm|m)\b", re.IGNORECASE)
_SIMPLE_DIM_CONTEXT_RE = re.compile(
    r"\b(hauteur|haut|largeur|longueur|profondeur|diametre|diagonale|ecran|tv|taille)\b",
    re.IGNORECASE,
)
_RESOLUTION_WIDTHS = {1024, 1280, 1366, 1600, 1920, 2560, 3840, 4096, 7680}
_RESOLUTION_HEIGHTS = {720, 768, 900, 1080, 1200, 1440, 2160, 4320}


def _to_float(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x.replace(",", "."))
    except ValueError:
        return None


def extraire_dimension(libelle: str) -> dict | None:
    """Extrait la dimension principale d'un libellé.

    Stratégie : on liste toutes les paires AxB(xC) trouvées et on garde celle
    dont la première dimension est la plus grande (= la dimension principale, pas une épaisseur).
    Unité : on cherche un cm/mm/inch dans la fenêtre autour ; sinon convention cm.
    """
    if not isinstance(libelle, str) or not libelle.strip():
        return None
    s = _strip_accents(libelle.lower())
    matches = list(_DIM_RE.finditer(s))
    if not matches:
        return None

    best = max(matches, key=lambda m: _to_float(m.group(1)) or 0)
    a, b, c = best.group(1), best.group(2), best.group(3)
    L = _to_float(a)
    l = _to_float(b)
    H = _to_float(c)

    start, end = best.span()
    window = s[max(0, start - 5): min(len(s), end + 6)]
    um = _UNIT_RE.search(window)
    unit = um.group(1).lower() if um else None

    if unit == "mm":
        L = (L or 0) / 10
        l = (l or 0) / 10
        if H is not None:
            H = H / 10
        unit = "cm"
    elif unit in ("inch", "pouce", "pouces", "po"):
        factor = 2.54
        L = (L or 0) * factor
        l = (l or 0) * factor
        if H is not None:
            H = H * factor
        unit = "cm"
    elif unit == "m":
        L = (L or 0) * 100
        l = (l or 0) * 100
        if H is not None:
            H = H * 100
        unit = "cm"
    elif unit == "cm":
        unit = "cm"
    else:
        unit = "cm (assumé)"

    label = f"{int(L) if L and L == int(L) else round(L, 1)}x{int(l) if l and l == int(l) else round(l, 1)}"
    if H is not None:
        label += f"x{int(H) if H and H == int(H) else round(H, 1)}"

    return {
        "L_cm": round(L, 2) if L is not None else None,
        "l_cm": round(l, 2) if l is not None else None,
        "H_cm": round(H, 2) if H is not None else None,
        "dim_label": label,
        "unite_detectee": unit,
    }


def extraire_dimensions_serie(libelles: pd.Series) -> pd.DataFrame:
    """Vectorisé : retourne un DataFrame indexé sur l'index de libelles."""
    cols = ["L_cm", "l_cm", "H_cm", "dim_label", "unite_detectee"]
    out = libelles.fillna("").astype(str).apply(extraire_dimension)
    rows = [item if isinstance(item, dict) else {} for item in out]
    return pd.DataFrame(rows, index=libelles.index).reindex(columns=cols)


def _format_dimension_number(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value == int(value) else str(round(value, 1))


def _repair_spaced_decimal_number(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\b(\d{1,4})\s+(\d{1,2})\b", r"\1.\2", value)


def _normalize_dimension_decimals(text: str) -> tuple[str, list[tuple[str, str]]]:
    repairs: list[tuple[str, str]] = []

    def repl(match: re.Match) -> str:
        raw = match.group("raw")
        if not re.search(r"\d\s+\d", raw):
            return raw
        normalized = _repair_spaced_decimal_number(raw) or raw
        if normalized != raw:
            repairs.append((raw, normalized))
        return normalized

    return _DIM_DECIMAL_SPACED_RE.sub(repl, text), repairs


def _to_float_v2(value: str | None) -> float | None:
    value = _repair_spaced_decimal_number(value)
    return _to_float(value)


def _convert_unit_v2(values: list[float | None], unit: str | None) -> tuple[list[float | None], str, str | None]:
    unit = (unit or "").lower()
    warning = None
    converted = values[:]
    if unit == "mm":
        converted = [None if v is None else v / 10 for v in values]
        unit_out = "cm"
    elif unit in ("inch", "pouce", "pouces", "po"):
        converted = [None if v is None else v * 2.54 for v in values]
        unit_out = "cm"
    elif unit == "m":
        numeric = [v for v in values if v is not None]
        if numeric and max(numeric) <= 20:
            converted = [None if v is None else v * 100 for v in values]
            unit_out = "cm"
        else:
            unit_out = "cm (assume)"
            warning = "unite_m_suspecte"
    elif unit == "cm":
        unit_out = "cm"
    else:
        unit_out = "cm (assume)"
    return converted, unit_out, warning


def _unit_after_dimension(text: str, end: int, values: list[float | None]) -> tuple[str | None, str | None]:
    after = text[end: min(len(text), end + 12)]
    if _ELECTRIC_UNIT_AFTER_RE.search(after):
        return None, "rejet_puissance"
    unit_match = _UNIT_AFTER_RE.search(after)
    if not unit_match:
        return None, None
    unit = unit_match.group(1).lower()
    if unit == "m":
        numeric = [v for v in values if v is not None]
        if numeric and max(numeric) > 20:
            return None, "unite_m_suspecte"
    return unit, None


def _dimension_warning_for_match(text: str, match: re.Match, values: list[float | None], unit: str | None) -> str | None:
    start, end = match.span()
    window = text[max(0, start - 30): min(len(text), end + 30)]
    if _POWER_DIM_RE.search(window) or _ELECTRIC_UNIT_AFTER_RE.search(text[end: min(len(text), end + 10)]):
        return "rejet_puissance"
    if len(values) >= 2 and values[0] is not None and values[1] is not None:
        first = int(round(values[0]))
        second = int(round(values[1]))
        if first in _RESOLUTION_WIDTHS and second in _RESOLUTION_HEIGHTS and (
            _RESOLUTION_WORDS_RE.search(window) or unit is None
        ):
            return "rejet_resolution"
        if unit is None and second >= 1000 and values[0] <= 100:
            return "rejet_reference"
    return None


def _empty_dimension_v2() -> dict:
    return {
        "L_cm": None,
        "l_cm": None,
        "H_cm": None,
        "dim_label": None,
        "unite_detectee": None,
        "diametre_cm": None,
        "dimension_simple_cm": None,
        "dimension_simple_type": None,
        "dimension_source": "non_detecte",
        "dimension_warning": None,
        "dimension_raw": None,
        "dimension_normalized": None,
        "nb_dimensions_detectees": 0,
        "dimensions_detectees_raw": None,
    }


def extraire_dimension_v2(libelle: str) -> dict:
    """Extrait les dimensions d'un libellé, avec colonnes d'audit et corrections prudentes."""
    result = _empty_dimension_v2()
    if not isinstance(libelle, str) or not libelle.strip():
        return result

    original = _strip_accents(libelle.lower())
    text, repairs = _normalize_dimension_decimals(original)
    matches = list(_DIM_RE.finditer(text))
    accepted = []
    rejected_warning = None
    for match in matches:
        values = [_to_float_v2(match.group(1)), _to_float_v2(match.group(2)), _to_float_v2(match.group(3))]
        unit, unit_warning = _unit_after_dimension(text, match.end(), values)
        warning = unit_warning or _dimension_warning_for_match(text, match, values, unit)
        if warning and warning.startswith("rejet_"):
            rejected_warning = rejected_warning or warning
            continue
        converted, unit_out, convert_warning = _convert_unit_v2(values, unit)
        value_warning = convert_warning
        numeric = [v for v in converted if v is not None]
        if numeric and max(numeric) > 1000:
            value_warning = value_warning or "valeur_suspecte"
        if len(converted) >= 2 and converted[0] is not None and converted[1] is not None:
            if min(converted[0], converted[1]) < 1:
                value_warning = value_warning or "valeur_suspecte"
        accepted.append(
            {
                "match": match,
                "values": converted,
                "unit": unit_out,
                "warning": value_warning,
                "source": "regex_normalisee_decimal_espace" if repairs else "regex_standard",
            }
        )

    if accepted:
        best = max(accepted, key=lambda item: item["values"][0] or 0)
        L, width, H = best["values"]
        label = f"{_format_dimension_number(L)}x{_format_dimension_number(width)}"
        if H is not None:
            label += f"x{_format_dimension_number(H)}"
        result.update(
            {
                "L_cm": round(L, 2) if L is not None else None,
                "l_cm": round(width, 2) if width is not None else None,
                "H_cm": round(H, 2) if H is not None else None,
                "dim_label": label,
                "unite_detectee": best["unit"],
                "dimension_source": best["source"],
                "dimension_warning": best["warning"] or ("decimal_space_repaired" if repairs else None),
                "dimension_raw": best["match"].group(0),
                "dimension_normalized": best["match"].group(0),
                "nb_dimensions_detectees": len(accepted),
                "dimensions_detectees_raw": " | ".join(item["match"].group(0) for item in accepted),
            }
        )
        if repairs:
            result["dimension_raw"] = " | ".join(raw for raw, _ in repairs)
            result["dimension_normalized"] = " | ".join(normalized for _, normalized in repairs)
        return result

    diam_match = _DIAM_H_RE.search(text) or _DIAM_RE.search(text)
    if diam_match:
        D = _to_float_v2(diam_match.group("D"))
        H = _to_float_v2(diam_match.groupdict().get("H"))
        unit = diam_match.groupdict().get("unit") or diam_match.groupdict().get("unit_h") or diam_match.groupdict().get("unit_d")
        converted, unit_out, warning = _convert_unit_v2([D, H], unit)
        D_cm, H_cm = converted[0], converted[1]
        result.update(
            {
                "H_cm": round(H_cm, 2) if H_cm is not None else None,
                "diametre_cm": round(D_cm, 2) if D_cm is not None else None,
                "dim_label": f"diam {_format_dimension_number(D_cm)}" + (f"xH{_format_dimension_number(H_cm)}" if H_cm is not None else ""),
                "unite_detectee": unit_out,
                "dimension_source": "diametre",
                "dimension_warning": warning,
                "dimension_raw": diam_match.group(0),
                "dimension_normalized": diam_match.group(0),
            }
        )
        return result

    simple_match = _SIMPLE_DIM_RE.search(text)
    if simple_match:
        start, end = simple_match.span()
        window = text[max(0, start - 25): min(len(text), end + 25)]
        value = _to_float_v2(simple_match.group("value"))
        converted, unit_out, warning = _convert_unit_v2([value], simple_match.group("unit"))
        context_match = _SIMPLE_DIM_CONTEXT_RE.search(window)
        result.update(
            {
                "dimension_simple_cm": round(converted[0], 2) if converted[0] is not None else None,
                "dimension_simple_type": context_match.group(1).strip() if context_match else "inconnu",
                "unite_detectee": unit_out,
                "dimension_source": "simple_cm",
                "dimension_warning": warning or rejected_warning,
                "dimension_raw": simple_match.group(0),
                "dimension_normalized": simple_match.group(0),
            }
        )
        return result

    if rejected_warning:
        result["dimension_warning"] = rejected_warning
    return result


def extraire_dimensions_v2_serie(libelles: pd.Series) -> pd.DataFrame:
    """Applique l'extraction de dimension V2 sur une série de libellés (colonnes d'audit incluses)."""
    cols = [
        "L_cm",
        "l_cm",
        "H_cm",
        "dim_label",
        "unite_detectee",
        "diametre_cm",
        "dimension_simple_cm",
        "dimension_simple_type",
        "dimension_source",
        "dimension_warning",
        "dimension_raw",
        "dimension_normalized",
        "nb_dimensions_detectees",
        "dimensions_detectees_raw",
    ]
    rows = libelles.fillna("").astype(str).apply(extraire_dimension_v2)
    return pd.DataFrame(list(rows), index=libelles.index).reindex(columns=cols)
