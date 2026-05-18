"""
Bot Value Betting Football — modèle de Poisson / Dixon-Coles.

Pipeline :
  1. Récupère les matchs à venir (API-Football) championnat par championnat.
  2. Estime lambda (buts attendus) via les forces attaque/défense domicile-extérieur.
  3. Ajuste selon les joueurs clés blessés ou suspendus.
  4. Calcule les probabilités 1/N/2 (Poisson + correction Dixon-Coles, exact).
  5. Compare aux cotes (The Odds API), détecte la value, mise via Kelly fractionné.
  6. Notifie sur Telegram et journalise les paris détectés.
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from scipy.stats import poisson

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("betbot")


# =============================================================================
# CONFIGURATION
# =============================================================================
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s invalide (%r) — valeur par défaut %s utilisée", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


# Secrets (variables d'environnement / .env)
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Paramètres
BANKROLL = _env_float("BANKROLL", 1000.0)
MIN_VALUE = _env_float("MIN_VALUE_PERCENT", 6.0) / 100.0   # edge minimal pour parier
KELLY_FRACTION = _env_float("KELLY_FRACTION", 0.25)        # quart de Kelly
MAX_BET_FRACTION = _env_float("MAX_BET_FRACTION", 0.05)    # plafond : 5 % de bankroll
MIN_BET = _env_float("MIN_BET", 5.0)                       # mise minimale viable
DAYS_AHEAD = _env_int("DAYS_AHEAD", 3)                     # fenêtre de matchs analysés
BOOKMAKER = os.getenv("BOOKMAKER", "unibet").lower()

# API-Football league id -> The Odds API sport key.
# À vérifier sur vos dashboards : un id/clé erroné renvoie simplement 0 cote
# pour ce championnat sans bloquer le reste du bot.
LEAGUES: dict[int, str] = {
    39: "soccer_epl",                      # Premier League (Angleterre)
    140: "soccer_spain_la_liga",           # La Liga (Espagne)
    78: "soccer_germany_bundesliga",       # Bundesliga (Allemagne)
    135: "soccer_italy_serie_a",           # Serie A (Italie)
    61: "soccer_france_ligue_one",         # Ligue 1 (France)
    88: "soccer_netherlands_eredivisie",   # Eredivisie (Pays-Bas)
    94: "soccer_portugal_primeira_liga",   # Primeira Liga (Portugal)
    40: "soccer_efl_champ",                # Championship (Angleterre)
    144: "soccer_belgium_first_div",       # Jupiler Pro League (Belgique)
    79: "soccer_germany_bundesliga2",      # 2. Bundesliga (Allemagne)
}

# Joueurs clés par id d'équipe API-Football. Sert à pénaliser lambda en cas
# d'absence (blessure/suspension). À tenir à jour. Les noms doivent correspondre
# à ceux renvoyés par l'API (accents compris).
KEY_PLAYERS: dict[int, list[str]] = {
    33: ["Erling Haaland", "Kevin De Bruyne", "Phil Foden"],            # Man City
    40: ["Mohamed Salah", "Darwin Núñez", "Luis Díaz"],                 # Liverpool
    50: ["Harry Kane", "Jamal Musiala", "Leroy Sané"],                  # Bayern
    529: ["Kylian Mbappé", "Vinicius Junior", "Jude Bellingham"],       # Real Madrid
    541: ["Robert Lewandowski", "Lamine Yamal", "Raphinha"],            # Barcelona
    489: ["Lautaro Martínez", "Marcus Thuram", "Nicolò Barella"],       # Inter
    492: ["Victor Osimhen", "Khvicha Kvaratskhelia", "Scott McTominay"],  # Napoli
    66: ["Ollie Watkins", "Morgan Rogers", "Emiliano Buendía"],         # Aston Villa
    34: ["Alexander Isak", "Anthony Gordon", "Bruno Guimarães"],        # Newcastle
}

# Modèle
MAX_GOALS = 10              # plafond de buts pour la grille de probabilités
DIXON_COLES_RHO = -0.05     # correction des scores faibles (corrélation 1-1, 0-0...)
HOME_LAMBDA_FLOOR = 0.2     # lambda minimal pour éviter les cas dégénérés
INJURY_PENALTY = 0.12       # réduction multiplicative de lambda par joueur clé absent

# API
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

# Cache persistant (réutilisé entre runs via actions/cache dans le workflow)
CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "stats.json"
CACHE_TTL = timedelta(hours=12)
BETS_LOG = Path("bets_log.csv")


# =============================================================================
# CACHE
# =============================================================================
def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("Cache illisible — réinitialisé")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except OSError as exc:
        log.warning("Écriture du cache impossible : %s", exc)


_CACHE: dict = _load_cache()


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry["ts"])
    except (KeyError, ValueError):
        return None
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    return entry["value"]


def _cache_set(key: str, value) -> None:
    _CACHE[key] = {"ts": datetime.now(timezone.utc).isoformat(), "value": value}


# =============================================================================
# COUCHE HTTP
# =============================================================================
def _request(url: str, *, headers=None, params=None, max_retries: int = 3, timeout: int = 10):
    """GET avec retry exponentiel. Renvoie le JSON décodé ou None."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("Requête échouée (%d/%d) : %s — pause %ds",
                        attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                log.warning("Réponse non-JSON depuis %s", url)
                return None
        if resp.status_code == 429:
            wait = 2 ** attempt
            log.warning("Rate limit (429) sur %s — pause %ds", url, wait)
            time.sleep(wait)
            continue
        log.warning("HTTP %s sur %s", resp.status_code, url)
        return None
    return None


def api_football(endpoint: str, params: dict | None = None) -> list:
    """Appel API-Football. Renvoie toujours une liste (vide en cas d'échec)."""
    data = _request(
        f"{API_FOOTBALL_BASE}{endpoint}",
        headers={"x-apisports-key": API_FOOTBALL_KEY},
        params=params or {},
    )
    if not isinstance(data, dict):
        return []
    return data.get("response") or []


# =============================================================================
# FORCE DES ÉQUIPES
# =============================================================================
@dataclass
class TeamStrength:
    attack_home: float   # buts marqués / match à domicile
    defense_home: float  # buts encaissés / match à domicile
    attack_away: float   # buts marqués / match à l'extérieur
    defense_away: float  # buts encaissés / match à l'extérieur


DEFAULT_STRENGTH = TeamStrength(attack_home=1.5, defense_home=1.1,
                                attack_away=1.1, defense_away=1.5)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_team_strength(league_id: int, team_id: int, season: int) -> TeamStrength:
    """Force offensive/défensive d'une équipe (avec cache)."""
    key = f"strength:{league_id}:{team_id}:{season}"
    cached = _cache_get(key)
    if cached is not None:
        return TeamStrength(**cached)

    data = _request(
        f"{API_FOOTBALL_BASE}/teams/statistics",
        headers={"x-apisports-key": API_FOOTBALL_KEY},
        params={"league": league_id, "team": team_id, "season": season},
    )
    resp = data.get("response") if isinstance(data, dict) else None
    if not resp:
        return DEFAULT_STRENGTH

    goals = resp.get("goals", {})

    def avg(side: str, where: str, fallback: float) -> float:
        # side = "for" | "against" ; where = "home" | "away"
        node = goals.get(side, {}).get("average", {})
        return _to_float(node.get(where)) or fallback

    strength = TeamStrength(
        attack_home=avg("for", "home", DEFAULT_STRENGTH.attack_home),
        defense_home=avg("against", "home", DEFAULT_STRENGTH.defense_home),
        attack_away=avg("for", "away", DEFAULT_STRENGTH.attack_away),
        defense_away=avg("against", "away", DEFAULT_STRENGTH.defense_away),
    )
    _cache_set(key, strength.__dict__)
    return strength


def expected_goals(home: TeamStrength, away: TeamStrength) -> tuple[float, float]:
    """Buts attendus en croisant l'attaque d'une équipe et la défense adverse."""
    lambda_home = (home.attack_home + away.defense_away) / 2
    lambda_away = (away.attack_away + home.defense_home) / 2
    return lambda_home, lambda_away


def key_players_out(team_id: int, season: int) -> int:
    """Nombre de joueurs clés blessés ou suspendus (0 si équipe non suivie)."""
    roster = KEY_PLAYERS.get(team_id)
    if not roster:
        return 0

    key = f"injuries:{team_id}:{season}"
    names = _cache_get(key)
    if names is None:
        resp = api_football("/injuries", {"team": team_id, "season": season})
        names = [item.get("player", {}).get("name", "") for item in resp]
        _cache_set(key, names)

    out = set(names)
    return sum(1 for player in roster if player in out)


# =============================================================================
# MODÈLE — PROBABILITÉS DE MATCH (Poisson + Dixon-Coles)
# =============================================================================
def _dixon_coles_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Correction Dixon-Coles de la dépendance des scores faibles."""
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    if i == 0 and j == 1:
        return 1 + lh * rho
    if i == 1 and j == 0:
        return 1 + la * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def match_probabilities(lambda_home: float, lambda_away: float) -> tuple[float, float, float]:
    """Probabilités exactes (1, N, 2) sur la grille des scores. Renvoie des valeurs normalisées."""
    lambda_home = max(lambda_home, HOME_LAMBDA_FLOOR)
    lambda_away = max(lambda_away, HOME_LAMBDA_FLOOR)

    home_pmf = [poisson.pmf(i, lambda_home) for i in range(MAX_GOALS + 1)]
    away_pmf = [poisson.pmf(j, lambda_away) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = total = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = (home_pmf[i] * away_pmf[j]
                 * _dixon_coles_tau(i, j, lambda_home, lambda_away, DIXON_COLES_RHO))
            total += p
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    if total <= 0:
        return 0.0, 0.0, 0.0
    return p_home / total, p_draw / total, p_away / total


# =============================================================================
# COTES
# =============================================================================
def get_odds(sport_key: str) -> list:
    """Cotes h2h d'un championnat via The Odds API."""
    data = _request(
        f"{ODDS_BASE}/{sport_key}/odds",
        params={
            "apiKey": THE_ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
    )
    return data if isinstance(data, list) else []


def normalize_name(name: str) -> str:
    """Normalise un nom d'équipe pour la comparaison inter-API."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    for token in (" fc", " cf", " afc", " sc", " ac ", " ssc", " calcio", "."):
        name = name.replace(token, " ")
    return " ".join(name.split())


def names_match(a: str, b: str) -> bool:
    """Rapprochement tolérant entre noms d'équipes de deux APIs différentes."""
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.6


def find_odds(events: list, home_name: str, away_name: str) -> tuple[float, float, float] | None:
    """Cotes (domicile, nul, extérieur) pour un match donné, ou None."""
    for event in events:
        ev_home = event.get("home_team", "")
        ev_away = event.get("away_team", "")
        if not (names_match(home_name, ev_home) and names_match(away_name, ev_away)):
            continue
        for bm in event.get("bookmakers", []):
            if BOOKMAKER not in bm.get("key", "").lower():
                continue
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                odd_home = prices.get(ev_home)
                odd_away = prices.get(ev_away)
                odd_draw = prices.get("Draw")
                if odd_home and odd_draw and odd_away:
                    return float(odd_home), float(odd_draw), float(odd_away)
    return None


def devig(odd_home: float, odd_draw: float, odd_away: float) -> tuple[float, float, float]:
    """Probabilités implicites du marché, marge bookmaker retirée."""
    inv = [1 / odd_home, 1 / odd_draw, 1 / odd_away]
    total = sum(inv)
    return tuple(x / total for x in inv)  # type: ignore[return-value]


# =============================================================================
# MISE — KELLY FRACTIONNÉ
# =============================================================================
def kelly_stake(prob: float, odds: float, bankroll: float) -> float:
    """Mise via Kelly fractionné. prob in [0, 1], odds = cote décimale.

    Renvoie 0 si Kelly déconseille le pari ou si la mise tombe sous le minimum.
    """
    net = odds - 1.0          # gain net par unité misée
    if net <= 0:
        return 0.0
    edge = net * prob - (1 - prob)   # == prob * odds - 1
    if edge <= 0:
        return 0.0
    kelly = edge / net
    fraction = min(kelly * KELLY_FRACTION, MAX_BET_FRACTION)
    stake = bankroll * fraction
    if stake < MIN_BET:
        return 0.0
    return round(stake, 2)


# =============================================================================
# TELEGRAM
# =============================================================================
def send_telegram(message: str) -> None:
    if not (BOT_TOKEN and CHAT_ID):
        log.warning("Telegram non configuré — message non envoyé")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.status_code != 200:
            log.error("Telegram a répondu %s : %s", resp.status_code, resp.text)
    except requests.RequestException as exc:
        log.error("Envoi Telegram échoué : %s", exc)


def build_message(bets: list[dict], season: int) -> str:
    if not bets:
        return (f"<b>ℹ️ Bot exécuté</b>\n"
                f"Aucun value bet ≥ {MIN_VALUE * 100:.0f}% trouvé.\n"
                f"Saison {season} — bankroll {BANKROLL:.0f}€.")

    total_stake = sum(b["stake"] for b in bets)
    lines = [f"<b>🔥 {len(bets)} VALUE BET(S)</b>\n"]
    for b in bets:
        match = html.escape(b["match"])
        lines.append(
            f"📅 <b>{match}</b>\n"
            f"   🎯 <b>{b['pari']}</b> @ {b['cote']} → <b>+{b['value_pct']}%</b>\n"
            f"   💰 Mise {b['stake']}€ | modèle {b['proba']}% / marché {b['market_prob']}%\n"
        )
    lines.append(f"💼 Bankroll {BANKROLL:.0f}€ | total engagé {total_stake:.0f}€")
    return "\n".join(lines)


# =============================================================================
# JOURNAL DES PARIS
# =============================================================================
def log_bets(bets: list[dict]) -> None:
    """Ajoute les paris détectés à un CSV pour permettre le suivi du ROI."""
    header = "timestamp,match,pari,cote,proba_modele,proba_marche,value_pct,mise\n"
    try:
        is_new = not BETS_LOG.exists()
        with BETS_LOG.open("a", encoding="utf-8") as f:
            if is_new:
                f.write(header)
            stamp = datetime.now(timezone.utc).isoformat()
            for b in bets:
                match = b["match"].replace(",", " ")
                f.write(f"{stamp},{match},{b['pari']},{b['cote']},"
                        f"{b['proba']},{b['market_prob']},{b['value_pct']},{b['stake']}\n")
    except OSError as exc:
        log.warning("Écriture du journal échouée : %s", exc)


# =============================================================================
# ANALYSE D'UN MATCH
# =============================================================================
def analyse_fixture(fixture: dict, league_id: int, season: int, odds_events: list) -> list[dict]:
    teams = fixture["teams"]
    home_name = teams["home"]["name"]
    away_name = teams["away"]["name"]
    home_id = teams["home"]["id"]
    away_id = teams["away"]["id"]

    odds = find_odds(odds_events, home_name, away_name)
    if not odds:
        return []
    odd_home, odd_draw, odd_away = odds

    home_str = get_team_strength(league_id, home_id, season)
    away_str = get_team_strength(league_id, away_id, season)
    lambda_home, lambda_away = expected_goals(home_str, away_str)

    # Pénalité multiplicative pour les joueurs clés absents.
    lambda_home *= (1 - INJURY_PENALTY) ** key_players_out(home_id, season)
    lambda_away *= (1 - INJURY_PENALTY) ** key_players_out(away_id, season)

    p_home, p_draw, p_away = match_probabilities(lambda_home, lambda_away)
    market = devig(odd_home, odd_draw, odd_away)

    bets: list[dict] = []
    for pari, prob, odd, mkt in (
        ("1", p_home, odd_home, market[0]),
        ("N", p_draw, odd_draw, market[1]),
        ("2", p_away, odd_away, market[2]),
    ):
        value = prob * odd - 1
        if value < MIN_VALUE:
            continue
        stake = kelly_stake(prob, odd, BANKROLL)
        if stake <= 0:
            continue
        bets.append({
            "match": f"{home_name} vs {away_name}",
            "pari": pari,
            "cote": round(odd, 2),
            "proba": round(prob * 100, 1),
            "market_prob": round(mkt * 100, 1),
            "value_pct": round(value * 100, 1),
            "stake": stake,
        })
    return bets


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def current_season(now: datetime) -> int:
    """Saison API-Football : l'année de début (août -> année courante)."""
    return now.year if now.month >= 8 else now.year - 1


def check_config() -> None:
    missing = [name for name, value in (
        ("API_FOOTBALL_KEY", API_FOOTBALL_KEY),
        ("THE_ODDS_API_KEY", THE_ODDS_API_KEY),
        ("BOT_TOKEN", BOT_TOKEN),
        ("CHAT_ID", CHAT_ID),
    ) if not value]
    if missing:
        log.error("Variables d'environnement manquantes : %s", ", ".join(missing))
        sys.exit(1)


def main() -> None:
    check_config()
    now = datetime.now(timezone.utc)
    season = current_season(now)
    log.info("Démarrage | bankroll=%.0f€ | seuil=%.0f%% | saison=%d",
             BANKROLL, MIN_VALUE * 100, season)

    date_from = now.date().isoformat()
    date_to = (now + timedelta(days=DAYS_AHEAD)).date().isoformat()

    all_bets: list[dict] = []
    for league_id, sport_key in LEAGUES.items():
        odds_events = get_odds(sport_key)
        if not odds_events:
            log.info("Ligue %d (%s) : aucune cote disponible", league_id, sport_key)
            continue

        fixtures = api_football("/fixtures", {
            "league": league_id,
            "season": season,
            "status": "NS",
            "from": date_from,
            "to": date_to,
        })
        log.info("Ligue %d : %d match(s), %d événement(s) de cotes",
                 league_id, len(fixtures), len(odds_events))

        for fixture in fixtures:
            try:
                all_bets.extend(analyse_fixture(fixture, league_id, season, odds_events))
            except Exception as exc:  # un match cassé ne doit pas tuer le run
                log.warning("Analyse d'un match échouée : %s", exc)

    _save_cache(_CACHE)
    all_bets.sort(key=lambda b: b["value_pct"], reverse=True)
    if all_bets:
        log_bets(all_bets)

    send_telegram(build_message(all_bets, season))
    log.info("Terminé : %d value bet(s) détecté(s)", len(all_bets))


if __name__ == "__main__":
    main()
