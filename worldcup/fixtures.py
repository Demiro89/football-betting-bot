"""Chargement des matchs à venir et de leurs cotes.

Format CSV attendu (`data/wc2026_fixtures.csv`) — éditable à la main ou
alimenté par une API (The Odds API, API-Football) :

    date,home_team,away_team,neutral,tournament,odd_1,odd_N,odd_2
    2026-06-11,Mexico,Spain,True,FIFA World Cup,3.40,3.30,2.15

  - `neutral` : True pour un match sur terrain neutre (cas général en CdM).
  - `tournament` : sert à pondérer l'importance (cf. config).
  - `odd_*` : cotes décimales du bookmaker (laisser vide si inconnues).

Les noms d'équipes doivent correspondre à ceux du jeu de données historique
(ex. « United States », « South Korea », « IR Iran »). `available_teams()` du
modèle aide à les retrouver côté interface.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_fixtures(path=None) -> pd.DataFrame:
    """Charge les matchs à venir. Renvoie un DataFrame (vide si fichier absent)."""
    path = path or config.FIXTURES_CSV
    if not path.exists():
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "neutral",
                     "tournament", "odd_1", "odd_N", "odd_2"]
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes", "oui"])
    else:
        df["neutral"] = True
    if "tournament" not in df.columns:
        df["tournament"] = "FIFA World Cup"
    for col in ("odd_1", "odd_N", "odd_2"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def odds_dict(row) -> dict[str, float]:
    """Extrait {'1','N','2': cote} d'une ligne de fixtures (cotes valides only).

    Accepte une Series pandas, un dict ou un namedtuple (itertuples).
    """
    if hasattr(row, "get"):           # Series ou dict
        getter = row.get
    elif hasattr(row, "_asdict"):     # namedtuple (itertuples)
        d = row._asdict()
        getter = d.get
    else:
        getter = lambda col: getattr(row, col, None)  # noqa: E731

    out: dict[str, float] = {}
    for sel, col in (("1", "odd_1"), ("N", "odd_N"), ("2", "odd_2")):
        val = getter(col)
        if val is not None and pd.notna(val) and float(val) > 1.0:
            out[sel] = float(val)
    return out
