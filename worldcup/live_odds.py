"""Cotes en temps réel + intelligence de marché (le cœur « anti-bookmaker »).

Trois leviers réels, utilisés par les parieurs professionnels :

1. LINE SHOPPING — pour chaque issue, on prend la **meilleure cote** disponible
   parmi TOUS les bookmakers interrogés. Parier systématiquement au meilleur prix
   augmente l'EV de plusieurs pourcents, sans aucun pari supplémentaire. C'est
   l'edge le plus fiable qui existe.

2. CONSENSUS DE MARCHÉ — on dévigue les cotes de chaque book puis on moyenne :
   on obtient une probabilité « consensus » qui approche la vraie probabilité
   bien mieux qu'un seul book. Détecter un book qui s'écarte du consensus
   (approche sharp/soft) est statistiquement la stratégie la plus rentable.

3. RÉSOLUTION DES NOMS — les noms d'équipes diffèrent entre l'API de cotes et le
   jeu de données historique ; on les rapproche par similarité pour que le
   modèle ML retrouve le bon Elo.

Tout passe par The Odds API (clé gratuite). Sans clé, le module reste importable
et l'app bascule sur le CSV d'exemple.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests

from . import config

log = logging.getLogger("worldcup.live_odds")

# Cache mémoire des réponses (clé = sport_key) pour ne pas brûler le quota API.
_ODDS_CACHE: dict[str, tuple[float, list]] = {}


# ---------------------------------------------------------------------------
# Couche HTTP
# ---------------------------------------------------------------------------
def _get(url: str, params: dict, max_retries: int = 3, timeout: int = 10):
    """GET JSON avec retry exponentiel. Renvoie l'objet décodé ou None."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("Cotes : requête échouée (%d/%d) : %s — pause %ds",
                        attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return None
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        log.warning("Cotes : HTTP %s sur %s", resp.status_code, url)
        return None
    return None


def fetch_events(sport_key: str, *, api_key: str | None = None,
                 regions: str | None = None, use_cache: bool = True) -> list:
    """Récupère les événements cotés (h2h) d'une compétition.

    Renvoie une liste d'événements bruts The Odds API (vide si rien/échec).
    """
    api_key = api_key or config.get_odds_api_key()
    regions = regions or config.ODDS_REGIONS
    if not api_key:
        return []

    if use_cache:
        cached = _ODDS_CACHE.get(sport_key)
        if cached and (time.time() - cached[0]) < config.ODDS_TTL_SECONDS:
            return cached[1]

    data = _get(
        f"{config.ODDS_API_BASE}/{sport_key}/odds",
        params={"apiKey": api_key, "regions": regions,
                "markets": "h2h", "oddsFormat": "decimal"},
    )
    events = data if isinstance(data, list) else []
    _ODDS_CACHE[sport_key] = (time.time(), events)
    return events


def fetch_all_events(sport_keys: list[str] | None = None, **kw) -> list:
    """Concatène les événements de plusieurs compétitions."""
    out: list = []
    for key in (sport_keys or config.SPORT_KEYS):
        events = fetch_events(key, **kw)
        for ev in events:
            ev["_sport_key"] = key
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Extraction des cotes 1/N/2 d'un événement
# ---------------------------------------------------------------------------
def _h2h_prices(bookmaker: dict, home: str, away: str) -> dict[str, float]:
    """Cotes 1/N/2 d'un bookmaker pour un événement (dict vide si marché absent)."""
    for market in bookmaker.get("markets", []):
        if market.get("key") != "h2h":
            continue
        prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
        out: dict[str, float] = {}
        if prices.get(home):
            out["1"] = float(prices[home])
        if prices.get("Draw"):
            out["N"] = float(prices["Draw"])
        if prices.get(away):
            out["2"] = float(prices[away])
        return out
    return {}


def best_odds(event: dict) -> dict[str, dict]:
    """LINE SHOPPING : meilleure cote par issue parmi tous les books.

    Renvoie {'1': {'odds': x, 'book': nom}, ...} pour les issues disponibles.
    """
    home, away = event.get("home_team", ""), event.get("away_team", "")
    best: dict[str, dict] = {}
    for bm in event.get("bookmakers", []):
        title = bm.get("title") or bm.get("key", "?")
        for sel, price in _h2h_prices(bm, home, away).items():
            if price <= 1.0:
                continue
            if sel not in best or price > best[sel]["odds"]:
                best[sel] = {"odds": round(price, 2), "book": title}
    return best


def consensus_probabilities(event: dict) -> dict[str, float]:
    """CONSENSUS : dévigue chaque book puis moyenne -> proba « marché ».

    Approche la vraie probabilité bien mieux qu'un seul bookmaker.
    """
    home, away = event.get("home_team", ""), event.get("away_team", "")
    acc = {"1": 0.0, "N": 0.0, "2": 0.0}
    n = 0
    for bm in event.get("bookmakers", []):
        prices = _h2h_prices(bm, home, away)
        if len(prices) < 3:                       # besoin des 3 issues pour dévig propre
            continue
        inv = {k: 1.0 / v for k, v in prices.items()}
        overround = sum(inv.values())
        for k in acc:
            acc[k] += inv[k] / overround
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in acc.items()}


def n_bookmakers(event: dict) -> int:
    return len(event.get("bookmakers", []))


# ---------------------------------------------------------------------------
# Rapprochement des noms d'équipes (cotes <-> dataset historique)
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = name.lower()
    for token in (" fc", " cf", " afc", " sc", " ac ", " ssc", " national team", "."):
        name = name.replace(token, " ")
    return " ".join(name.split())


# Alias fréquents entre The Odds API et le dataset martj42.
_TEAM_ALIASES = {
    "usa": "united states",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "iran": "ir iran",
    "ivory coast": "cote d'ivoire",
    "czech republic": "czechia",
    "china": "china pr",
    "cape verde": "cabo verde",
}


def resolve_team(name: str, known_teams: list[str], threshold: float = 0.6) -> str:
    """Rapproche un nom de l'API du nom utilisé par le modèle (sinon renvoie tel quel)."""
    target = normalize_name(name)
    target = _TEAM_ALIASES.get(target, target)

    norm_known = {normalize_name(t): t for t in known_teams}
    if target in norm_known:
        return norm_known[target]

    best_team, best_ratio = name, 0.0
    for nt, original in norm_known.items():
        if target and (target in nt or nt in target):
            return original
        ratio = SequenceMatcher(None, target, nt).ratio()
        if ratio > best_ratio:
            best_team, best_ratio = original, ratio
    return best_team if best_ratio >= threshold else name


# ---------------------------------------------------------------------------
# Conversion en lignes de « fixtures » exploitables par l'app
# ---------------------------------------------------------------------------
def event_to_fixture(event: dict, known_teams: list[str]) -> dict | None:
    """Transforme un événement coté en ligne de fixture enrichie.

    Inclut : meilleures cotes (line shopping), consensus marché, nb de books,
    et les noms d'équipes résolus pour le modèle.
    """
    home_raw, away_raw = event.get("home_team"), event.get("away_team")
    if not home_raw or not away_raw:
        return None
    best = best_odds(event)
    if not best:
        return None

    commence = event.get("commence_time")
    try:
        date = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        date = datetime.now(timezone.utc)

    sport_key = event.get("_sport_key", "")
    tournament = "FIFA World Cup" if "world_cup" in sport_key else sport_key

    return {
        "date": date,
        "home_team": resolve_team(home_raw, known_teams),
        "away_team": resolve_team(away_raw, known_teams),
        "home_team_raw": home_raw,
        "away_team_raw": away_raw,
        # En club/championnat le terrain n'est pas neutre ; en CdM il l'est.
        "neutral": "world_cup" in sport_key or "championship" in sport_key,
        "tournament": tournament,
        "sport_key": sport_key,
        "best_odds": best,
        "consensus": consensus_probabilities(event),
        "n_books": n_bookmakers(event),
        "odd_1": best.get("1", {}).get("odds"),
        "odd_N": best.get("N", {}).get("odds"),
        "odd_2": best.get("2", {}).get("odds"),
    }


def live_fixtures(known_teams: list[str], sport_keys: list[str] | None = None, **kw) -> list[dict]:
    """Liste des matchs cotés en direct, enrichis et prêts pour le modèle."""
    events = fetch_all_events(sport_keys, **kw)
    out = []
    for ev in events:
        row = event_to_fixture(ev, known_teams)
        if row:
            out.append(row)
    out.sort(key=lambda r: r["date"])
    return out
