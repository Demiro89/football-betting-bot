#!/usr/bin/env python3
"""ÉTAPE 4 — Application web Streamlit (temps réel, déployable sur Streamlit Cloud).

Lancement local :
    pip install -r requirements.txt
    export THE_ODDS_API_KEY=ta_cle       # cotes en direct (optionnel), ou .env
    streamlit run app.py

Déploiement Streamlit Cloud : pousser le dépôt, créer l'app sur
share.streamlit.io (fichier principal = app.py), renseigner THE_ODDS_API_KEY
dans les Secrets. Le modèle s'entraîne automatiquement au premier lancement
(aucune étape manuelle).

Fonctionnalités :
  - Cotes EN DIRECT multi-bookmakers (The Odds API) avec LINE SHOPPING + auto-refresh.
  - Prédictions 1/N/2 : modèle ML, consensus de marché, ou ensemble des deux.
  - Alertes « Value Bets » mises en évidence + mise Kelly recommandée.
  - Suivi CLV (Closing Line Value) et ROI réel des paris journalisés.

⚠️ Les paris comportent un risque de perte. Aucun gain n'est garanti.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import streamlit as st

# Pont secrets -> variables d'environnement, AVANT d'importer worldcup (config
# lit les variables d'env à l'import). Permet d'utiliser les Secrets Streamlit
# Cloud sans rien committer.
try:
    for _k in ("THE_ODDS_API_KEY", "SPORT_KEYS", "ODDS_REGIONS", "BANKROLL",
               "MIN_VALUE_PERCENT", "KELLY_FRACTION", "ENSEMBLE_MARKET_WEIGHT"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass  # pas de secrets.toml en local : on utilise .env / l'environnement

import pandas as pd  # noqa: E402

from worldcup import config, data, live_odds, simulator, tickets, tracking  # noqa: E402
from worldcup.fixtures import load_fixtures, odds_dict  # noqa: E402
from worldcup.model import MatchPredictor  # noqa: E402
from worldcup.value import (  # noqa: E402
    ensemble_probabilities,
    find_value_bets,
    implied_probabilities,
)

st.set_page_config(page_title="Coupe du Monde — Value Bets ML", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Chargement / auto-entraînement du modèle (mis en cache pour la session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model() -> MatchPredictor:
    """Charge le modèle pré-entraîné livré dans le dépôt (chargement instantané).

    Repli : s'il est absent, entraîne à la volée. Le modèle versionné
    (models/wc_model.joblib) évite tout calcul lourd au démarrage du cloud.
    """
    if config.MODEL_FILE.exists():
        return MatchPredictor.load()
    # Repli (modèle non livré) : entraînement à la volée.
    with st.spinner("Premier lancement : entraînement du modèle "
                    "(téléchargement des données + apprentissage)…"):
        df = data.load_results()
        predictor = MatchPredictor().fit(df, calibrate=True)
        try:
            predictor.save()
        except OSError:
            pass  # système de fichiers en lecture seule : modèle gardé en mémoire
    return predictor


# Chargement protégé : en cas d'échec, on affiche l'erreur au lieu d'un écran blanc.
try:
    _MODEL = get_model()
except Exception as _exc:  # noqa: BLE001
    st.error("❌ Impossible de charger le modèle de prédiction.\n\n"
             f"Détail technique : `{type(_exc).__name__}: {_exc}`\n\n"
             "Si le problème persiste, ouvre **Manage app → logs** et envoie "
             "le message d'erreur.")
    st.stop()


def predictions_for(model, home, away, date, neutral, tournament, consensus, market_weight):
    """Renvoie (probs_modele, probs_finales) selon la source de proba choisie."""
    model_probs = model.predict_match(home, away, date, neutral, tournament)
    final = ensemble_probabilities(model_probs, consensus, market_weight)
    return model_probs, final


def render_match_card(title, subtitle, final_probs, model_probs, odds, books, bets,
                      match_key, reliable=True):
    """Affiche une carte de match avec prédictions, cotes et value bets."""
    with st.container(border=True):
        st.subheader(title)
        st.caption(subtitle)
        if not reliable:
            st.warning("⚠️ Équipe(s) non reconnue(s) par le modèle (entraîné sur les "
                       "sélections nationales). Prédiction **non fiable** — aucun pari "
                       "proposé sur ce match.")
        c1, cn, c2 = st.columns(3)
        c1.metric("1 — Domicile", f"{final_probs['1']*100:.0f}%",
                  help=f"Modèle ML seul : {model_probs['1']*100:.0f}%")
        cn.metric("N — Nul", f"{final_probs['N']*100:.0f}%",
                  help=f"Modèle ML seul : {model_probs['N']*100:.0f}%")
        c2.metric("2 — Extérieur", f"{final_probs['2']*100:.0f}%",
                  help=f"Modèle ML seul : {model_probs['2']*100:.0f}%")
        if odds:
            oc = st.columns(3)
            for i, sel in enumerate(("1", "N", "2")):
                book = books.get(sel, {}).get("book", "")
                val = odds.get(sel)
                oc[i].write(f"Cote {sel} : **{val if val else '—'}**"
                            + (f"  \n*{book}*" if book else ""))
        for j, b in enumerate(bets):
            st.success(
                f"💡 **Value — {b.label}** @ {b.odds} "
                f"({books.get(b.selection, {}).get('book', 'meilleur prix')}) · "
                f"value **+{b.edge_pct:.1f}%** · mise **{b.stake:.2f} €** "
                f"(proba retenue {b.model_prob*100:.0f}% vs marché {b.market_prob*100:.0f}%)"
            )
            if st.button("📌 Suivre ce pari (CLV/ROI)", key=f"log_{match_key}_{j}"):
                tracking.log_bet(title, b.label, b.odds, b.model_prob,
                                 b.market_prob, b.edge_pct, b.stake)
                st.toast(f"Pari enregistré : {b.label} @ {b.odds}", icon="✅")


# ---------------------------------------------------------------------------
# Barre latérale — paramètres
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Paramètres")

_has_key = bool(config.get_odds_api_key())
source = st.sidebar.radio("Source des matchs", ["🔴 Cotes en direct (API)", "📄 CSV local"],
                          index=0 if _has_key else 1)
live_mode = source.startswith("🔴")

prob_source = st.sidebar.selectbox(
    "Probabilité retenue pour la value",
    ["Ensemble (ML + marché)", "Modèle ML seul", "Consensus marché seul"],
)
market_weight = config.ENSEMBLE_MARKET_WEIGHT
if prob_source == "Ensemble (ML + marché)":
    market_weight = st.sidebar.slider("Poids du marché dans l'ensemble", 0.0, 1.0,
                                      float(config.ENSEMBLE_MARKET_WEIGHT), 0.05)
elif prob_source == "Modèle ML seul":
    market_weight = 0.0
elif prob_source == "Consensus marché seul":
    market_weight = 1.0

bankroll = st.sidebar.number_input("Bankroll (€)", min_value=10.0, value=float(config.BANKROLL),
                                   step=10.0)
min_value_pct = st.sidebar.slider("Seuil de value minimal (%)", 0, 30,
                                  int(config.MIN_VALUE * 100), 1)

auto_refresh = st.sidebar.toggle("Auto-refresh (temps réel)", value=live_mode)
refresh_secs = st.sidebar.slider("Intervalle de refresh (s)", 15, 180, 60, 15,
                                 disabled=not auto_refresh)

st.sidebar.caption(
    f"Kelly fractionné : {config.KELLY_FRACTION:.0%} du Kelly, plafonné à "
    f"{config.MAX_BET_FRACTION:.0%} de la bankroll."
)
st.sidebar.divider()
st.sidebar.info("Pariez de façon responsable : uniquement de l'argent que vous "
                "pouvez perdre. Aucun modèle ne garantit de gain.")

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.title("⚽ Coupe du Monde — Détecteur de Value Bets (ML + temps réel)")
st.caption("XGBoost calibré · line shopping multi-books · consensus de marché · gestion Kelly")

model = _MODEL
meta = model.metadata
# Sélections connues du modèle (entraîné sur les matchs internationaux) :
# sert à écarter les équipes non reconnues (clubs, noms non rapprochés).
KNOWN_TEAMS = set(model.builder.known_teams())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Moteur", meta.get("backend", "?"))
m2.metric("Matchs d'entraînement", f"{meta.get('n_train', 0):,}".replace(",", " "))
m3.metric("Calibré", "Oui" if meta.get("calibrated") else "Non")
m4.metric("Données jusqu'au", meta.get("last_match_date", "?"))

if live_mode and not config.get_odds_api_key():
    st.warning("`THE_ODDS_API_KEY` non définie — bascule sur le CSV local. "
               "Ajoutez la clé (gratuite sur the-odds-api.com) dans les Secrets "
               "Streamlit Cloud (ou .env en local) pour les cotes en direct.")
    live_mode = False


# ---------------------------------------------------------------------------
# Construction des blocs de matchs (live ou CSV)
# ---------------------------------------------------------------------------
def build_blocks() -> list[dict]:
    blocks: list[dict] = []
    if live_mode:
        rows = live_odds.live_fixtures(model.builder.known_teams())
        for r in rows:
            odds = {k: r[f"odd_{k}"] for k in ("1", "N", "2") if r.get(f"odd_{k}")}
            # Garde-fou : le modèle n'a appris QUE sur les sélections nationales.
            # Si une équipe n'est pas reconnue (club, nom non rapproché), on
            # n'invente AUCUN pari : prédiction non fiable.
            reliable = (r["home_team"] in KNOWN_TEAMS and r["away_team"] in KNOWN_TEAMS)
            model_probs, final = predictions_for(
                model, r["home_team"], r["away_team"], pd.Timestamp(r["date"]),
                r["neutral"], r["tournament"], r.get("consensus"), market_weight)
            bets = find_value_bets(final, odds, bankroll=bankroll,
                                   min_value=min_value_pct / 100.0) if reliable else []
            blocks.append({
                "title": f"{r['home_team_raw']} vs {r['away_team_raw']}",
                "subtitle": f"{pd.Timestamp(r['date']):%d/%m %H:%M} · {r['tournament']} · "
                            f"{r['n_books']} books · meilleures cotes",
                "final": final, "model": model_probs, "odds": odds,
                "books": r["best_odds"], "bets": bets, "reliable": reliable,
                "key": f"{r['home_team_raw']}_{r['away_team_raw']}",
            })
    else:
        fixtures = load_fixtures()
        for r in fixtures.itertuples(index=False):
            odds = odds_dict(r)
            # En CSV : pas de multi-books ; on dérive un « consensus » dévigé du book unique.
            consensus = implied_probabilities(odds) or None
            reliable = (r.home_team in KNOWN_TEAMS and r.away_team in KNOWN_TEAMS)
            model_probs, final = predictions_for(
                model, r.home_team, r.away_team, pd.Timestamp(r.date),
                bool(r.neutral), r.tournament, consensus, market_weight)
            bets = find_value_bets(final, odds, bankroll=bankroll,
                                   min_value=min_value_pct / 100.0) if reliable else []
            blocks.append({
                "title": f"{r.home_team} vs {r.away_team}",
                "subtitle": f"{pd.Timestamp(r.date):%d/%m/%Y} · {r.tournament} · "
                            f"{'terrain neutre' if bool(r.neutral) else 'à domicile'}",
                "final": final, "model": model_probs, "odds": odds,
                "books": {}, "bets": bets, "reliable": reliable,
                "key": f"{r.home_team}_{r.away_team}",
            })
    return blocks


def render_tracking() -> None:
    """Onglet suivi CLV / ROI (recalculé à chaque rafraîchissement)."""
    st.subheader("Closing Line Value & ROI réel")
    st.caption("Le CLV (cote prise vs cote de clôture) est le meilleur indicateur "
               "AVANT d'avoir un gros échantillon : un CLV moyen positif = vous battez "
               "le marché. Renseignez `closing_odds` et `result` dans le CSV pour l'affiner.")
    s = tracking.summary()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Paris suivis", s["n_total"])
    k2.metric("CLV moyen", f"{s['avg_clv_pct']:+.2f}%" if s["avg_clv_pct"] is not None else "—",
              help="Basé sur les paris dont la cote de clôture est renseignée.")
    k3.metric("Paris réglés", s["n_settled"])
    k4.metric("ROI réel", f"{s['roi_pct']:+.1f}%" if s["roi_pct"] is not None else "—")

    bets = tracking.load_bets()
    if bets:
        st.dataframe(pd.DataFrame(bets), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Télécharger le journal (CSV)",
                           data=config.BETS_LOG.read_text(encoding="utf-8"),
                           file_name="tracked_bets.csv", mime="text/csv")
    else:
        st.info("Aucun pari suivi. Cliquez « 📌 Suivre ce pari » sur une value bet.")
    if s["n_settled"] < 30:
        st.warning("Moins de 30 paris réglés : le ROI n'est pas encore significatif. "
                   "Pilotez d'abord sur le CLV.")


def _value_legs(blocks) -> list[tickets.TicketLeg]:
    """Jambes « value » (issues rentables) extraites de tous les matchs."""
    legs = []
    for blk in blocks:
        for b in blk["bets"]:
            legs.append(tickets.TicketLeg(
                match=blk["title"], selection=b.selection, label=b.label,
                odds=b.odds, model_prob=b.model_prob, market_prob=b.market_prob))
    return legs


def _all_legs(blocks) -> dict[str, tickets.TicketLeg]:
    """Toutes les issues cotées (pour construire un ticket à la main)."""
    out: dict[str, tickets.TicketLeg] = {}
    for blk in blocks:
        if not blk.get("reliable", True):
            continue  # pas d'équipe inconnue dans le constructeur
        market = implied_probabilities(blk["odds"])
        for sel in ("1", "N", "2"):
            odd = blk["odds"].get(sel)
            if not odd:
                continue
            label = f"{blk['title']} — {config.OUTCOME_LABELS[sel]} @ {odd}"
            out[label] = tickets.TicketLeg(
                match=blk["title"], selection=sel, label=config.OUTCOME_LABELS[sel],
                odds=float(odd), model_prob=float(blk["final"].get(sel, 0.0)),
                market_prob=float(market.get(sel, 1.0 / float(odd))))
    return out


def _ticket_signature(t: tickets.Ticket) -> str:
    """Identifiant stable d'un ticket (pour la clé de bouton Streamlit)."""
    return "|".join(f"{leg.match}-{leg.selection}" for leg in t.legs)


def _render_ticket(t: tickets.Ticket, key_prefix: str = "ticket") -> None:
    """Affiche une carte de ticket (simple ou combiné) + bouton de suivi."""
    legs_txt = "  +  ".join(f"**{leg.match} : {leg.label}** @ {leg.odds}" for leg in t.legs)
    with st.container(border=True):
        st.markdown(f"🎟️ **Ticket {t.kind}** ({len(t.legs)} sélection·s)")
        st.markdown(legs_txt)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cote totale", f"{t.combined_odds:.2f}")
        c2.metric("Proba modèle", f"{t.model_prob*100:.0f}%")
        c3.metric("Value", f"{t.edge_pct:+.1f}%")
        c4.metric("Mise conseillée", f"{t.stake:.2f} €")
        st.caption(f"Gain brut potentiel : **{t.potential_return:.2f} €** "
                   f"(mise {t.stake:.2f} € × cote {t.combined_odds:.2f}) · "
                   f"proba marché {t.market_prob*100:.1f}%")
        if st.button("📌 Suivre ce ticket (CLV/ROI)",
                     key=f"track_{key_prefix}_{_ticket_signature(t)}"):
            tracking.log_ticket(t)
            st.toast(f"Ticket {t.kind} enregistré (cote {t.combined_odds:.2f})", icon="✅")


def _confidence_legs(blocks, min_conf: float):
    """Pour chaque match fiable, la sélection la plus probable selon le modèle,
    si sa proba >= seuil et qu'une cote existe. Triées par confiance décroissante."""
    picks = []
    for blk in blocks:
        if not blk.get("reliable", True) or not blk["odds"]:
            continue
        market = implied_probabilities(blk["odds"])
        # Issue la plus probable PARMI celles qui ont une cote.
        best_sel = max((s for s in ("1", "N", "2") if blk["odds"].get(s)),
                       key=lambda s: blk["final"].get(s, 0.0), default=None)
        if best_sel is None:
            continue
        p = blk["final"].get(best_sel, 0.0)
        if p < min_conf:
            continue
        picks.append(tickets.TicketLeg(
            match=blk["title"], selection=best_sel,
            label=config.OUTCOME_LABELS[best_sel], odds=float(blk["odds"][best_sel]),
            model_prob=float(p), market_prob=float(market.get(best_sel, 0.0))))
    picks.sort(key=lambda leg: leg.model_prob, reverse=True)
    return picks


def render_confidence_coupon(blocks, bankroll: float) -> None:
    """Coupon « haute confiance » : matchs où le modèle est le plus sûr (live)."""
    st.subheader("🎯 Coupon haute confiance (taux de réussite estimé élevé)")
    st.caption("Les sélections où le modèle est le PLUS sûr, en direct. "
               "⚠️ Taux de réussite élevé ≠ rentable : ce sont souvent des favoris "
               "à petite cote. Et combiner FAIT BAISSER le taux de réussite.")
    c1, c2 = st.columns(2)
    min_conf = c1.slider("Confiance minimale (%)", 50, 90, 65, 5,
                         key="conf_min") / 100.0
    n_legs = c2.slider("Matchs dans le combiné", 2, 6, 3, 1, key="conf_legs")

    picks = _confidence_legs(blocks, min_conf)
    if not picks:
        st.info(f"Aucune sélection ≥ {min_conf*100:.0f}% de confiance sur les matchs "
                "cotés actuellement (équipes reconnues). Baisse le seuil ou reviens plus tard.")
        return

    st.markdown(f"**{len(picks)} sélection·s** au-dessus de {min_conf*100:.0f}% de confiance :")
    st.dataframe(pd.DataFrame([{
        "Match": leg.match, "Pari": leg.label, "Cote": leg.odds,
        "Taux de réussite estimé": f"{leg.model_prob*100:.0f}%",
        "Value": f"{(leg.model_prob*leg.odds-1)*100:+.1f}%",
    } for leg in picks]), use_container_width=True, hide_index=True)

    chosen = picks[:n_legs]
    if len(chosen) >= 2:
        combo = tickets.build_ticket(chosen, bankroll=bankroll, kind="combiné")
        success = combo.model_prob * 100
        st.markdown(f"#### 🎫 Combiné des {len(chosen)} plus sûrs")
        st.warning(f"Taux de réussite estimé du combiné : **{success:.0f}%** "
                   f"(produit des {len(chosen)} probas — il faut que TOUS passent). "
                   f"Cote totale **{combo.combined_odds:.2f}**.")
        _render_ticket(combo, key_prefix="confidence")
        st.caption("Compare bien : 1 seul pari = taux de réussite le plus haut. "
                   "Chaque match ajouté multiplie le risque.")


def render_tickets(blocks, bankroll: float) -> None:
    """Onglet Tickets : coupon haute confiance + propositions value + constructeur."""
    render_confidence_coupon(blocks, bankroll)
    st.divider()
    legs = _value_legs(blocks)

    st.subheader("🎯 Tickets simples recommandés")
    st.caption("Approche la plus rentable : une sélection value = un ticket. "
               "Triés par value décroissante.")
    proposals = tickets.propose_tickets(legs, bankroll=bankroll)
    if proposals["singles"]:
        for i, t in enumerate(proposals["singles"]):
            _render_ticket(t, key_prefix=f"single{i}")
    else:
        st.info("Aucune sélection value pour le moment → aucun ticket simple à proposer.")

    st.divider()
    st.subheader("🎲 Combiné « value » proposé (optionnel, plus risqué)")
    if proposals["combo"]:
        st.warning("Un combiné multiplie la marge du bookmaker ET la variance. "
                   "Mise volontairement réduite. À ne jouer que ponctuellement.")
        _render_ticket(proposals["combo"], key_prefix="combo")
        if proposals["combo"].stake <= 0:
            st.caption("ℹ️ Kelly conseille ici une mise inférieure au minimum : "
                       "combiné « pour le plaisir », à ne jouer qu'avec une toute petite somme.")
    else:
        st.info("Pas de combiné value pertinent (il faut au moins 2 sélections value "
                "sur des matchs différents).")

    st.divider()
    st.subheader("🛠️ Construire mon propre ticket")
    options = _all_legs(blocks)
    if not options:
        st.info("Aucune issue cotée disponible actuellement.")
        return
    chosen_labels = st.multiselect(
        "Choisis tes sélections (matchs différents recommandés) :",
        options=list(options.keys()), key="ticket_builder")
    if chosen_labels:
        chosen = [options[lbl] for lbl in chosen_labels]
        matches = [leg.match for leg in chosen]
        if len(set(matches)) < len(matches):
            st.error("⚠️ Tu as choisi plusieurs issues d'un même match : ce n'est pas "
                     "combinable (résultats corrélés / impossible chez le bookmaker).")
        else:
            t = tickets.build_ticket(chosen, bankroll=bankroll)
            _render_ticket(t, key_prefix="builder")
            if t.edge_pct <= 0:
                st.warning("Ce ticket est à value négative : le modèle l'estime "
                           "perdant sur la durée. La mise conseillée est 0.")


def _top_teams_by_elo(n: int = 16) -> list[str]:
    """Les n meilleures sélections selon l'Elo du modèle (défaut du simulateur)."""
    ranked = sorted(KNOWN_TEAMS, key=lambda t: model.builder.elo_of(t), reverse=True)
    return ranked[:n]


def render_simulator() -> None:
    """Onglet simulateur de tournoi « façon Opta » (Monte-Carlo)."""
    st.subheader("🏆 Simulateur de tournoi (méthode façon Opta)")
    st.caption("On rejoue le tournoi des milliers de fois selon les forces du modèle "
               "(terrain neutre, tirage aléatoire). Résultat : la probabilité, pour "
               "chaque nation, d'atteindre chaque stade. Indicatif — basé sur notre modèle, "
               "pas sur les données privées d'Opta.")

    default = _top_teams_by_elo(16)
    chosen = st.multiselect(
        "Équipes en lice (choisis 8, 16 ou 32 pour un tableau parfait) :",
        options=sorted(KNOWN_TEAMS), default=default, key="sim_teams")
    n_sims = st.select_slider("Nombre de simulations",
                              options=[1000, 2000, 5000, 10000, 20000],
                              value=5000, key="sim_n")

    if st.button("🎲 Lancer la simulation", key="sim_run", type="primary"):
        if len(chosen) < 2:
            st.warning("Choisis au moins 2 équipes.")
        else:
            with st.spinner(f"Simulation de {n_sims:,} tournois…".replace(",", " ")):
                res = simulator.simulate_from_model(model, chosen, n_sims=n_sims)
            rows = [{"Équipe": t,
                     "Champion": v["champion"], "Finale": v["finale"],
                     "Demi": v["demi"], "Quart": v["quart"]}
                    for t, v in res.items()]
            df = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
            st.session_state["sim_result"] = df

    df = st.session_state.get("sim_result")
    if df is not None:
        st.markdown("#### 🥇 Probabilité de remporter le tournoi")
        st.bar_chart(df.set_index("Équipe")["Champion"], height=320)
        show = df.copy()
        for col in ("Champion", "Finale", "Demi", "Quart"):
            show[col] = (show[col] * 100).map(lambda x: f"{x:.1f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("« Champion » = remporte le tournoi · « Finale/Demi/Quart » = "
                   "atteint au moins ce stade. Les colonnes Champion somment à ~100 %.")
    else:
        st.info("Choisis tes équipes puis clique « Lancer la simulation ».")


@st.fragment(run_every=(refresh_secs if auto_refresh else None))
def render_board():
    """Tableau de bord rafraîchi automatiquement en mode temps réel.

    Le fragment crée ses propres onglets à chaque exécution : l'auto-refresh
    remplace proprement le contenu sans le dupliquer.
    """
    blocks = build_blocks()
    st.caption(f"Dernière mise à jour : {datetime.now(timezone.utc):%H:%M:%S UTC} · "
               f"{len(blocks)} match(s)" + (" · 🔴 LIVE" if live_mode else ""))

    tab_value, tab_tickets, tab_sim, tab_matches, tab_track = st.tabs(
        ["🔥 Value Bets", "🎟️ Tickets", "🏆 Simulateur",
         "📋 Tous les matchs", "📈 Suivi CLV / ROI"])

    all_bets = []
    for blk in blocks:
        for b in blk["bets"]:
            all_bets.append({
                "Match": blk["title"], "Pari": b.label, "Cote": b.odds,
                "Book": blk["books"].get(b.selection, {}).get("book", "—"),
                "Proba retenue": f"{b.model_prob*100:.0f}%",
                "Proba marché": f"{b.market_prob*100:.0f}%",
                "Value": f"+{b.edge_pct:.1f}%", "Mise (€)": b.stake,
            })

    with tab_value:
        if not blocks:
            st.info("Aucun match coté pour le moment (hors-saison ou clé/quota). "
                    "Réessayez plus tard ou basculez sur le CSV local.")
        elif all_bets:
            total = sum(b["Mise (€)"] for b in all_bets)
            st.success(f"**{len(all_bets)} value bet(s)** · total engagé conseillé : "
                       f"**{total:.2f} €** ({total/bankroll*100:.1f}% de la bankroll)")
            st.dataframe(pd.DataFrame(all_bets), use_container_width=True, hide_index=True)
            st.caption("Value = proba retenue × meilleure cote − 1. Mise = Kelly fractionné. "
                       "« Book » indique où obtenir le meilleur prix (line shopping).")
        else:
            st.info(f"Aucun value bet ≥ {min_value_pct}% actuellement. "
                    "C'est normal : la plupart des matchs n'offrent pas de value.")

    with tab_tickets:
        if not blocks:
            st.info("Aucun match coté pour le moment.")
        else:
            render_tickets(blocks, bankroll)

    with tab_sim:
        render_simulator()

    with tab_matches:
        for blk in blocks:
            flag = "🔥 " if blk["bets"] else ("⚠️ " if not blk.get("reliable", True) else "")
            render_match_card(flag + blk["title"], blk["subtitle"], blk["final"],
                              blk["model"], blk["odds"], blk["books"], blk["bets"],
                              blk["key"], reliable=blk.get("reliable", True))

    with tab_track:
        render_tracking()


render_board()

st.divider()
st.caption(
    f"Généré le {datetime.now(timezone.utc):%d/%m/%Y %H:%M UTC}. "
    "⚠️ Les paris sportifs comportent un risque de perte. Suivez votre CLV/ROI et "
    "arrêtez si le modèle perd de l'argent."
)
