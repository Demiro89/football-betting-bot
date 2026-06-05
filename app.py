#!/usr/bin/env python3
"""ÉTAPE 4 — Interface web Streamlit.

Lancement :
    pip install -r requirements-ml.txt
    python train.py            # entraîne et sauvegarde le modèle (une fois)
    streamlit run app.py

Affiche :
  - la liste des prochains matchs de Coupe du Monde ;
  - les prédictions 1 / N / 2 en pourcentage ;
  - les alertes « Value Bets » mises en évidence ;
  - la mise recommandée (Kelly fractionné) selon la bankroll saisie.

Avertissement intégré : les paris comportent un risque de perte. L'outil aide
à la décision, il ne garantit aucun gain.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from worldcup import config
from worldcup.fixtures import load_fixtures, odds_dict
from worldcup.model import MatchPredictor
from worldcup.value import find_value_bets

st.set_page_config(page_title="Coupe du Monde — Value Bets ML", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Chargement (mis en cache pour ne pas recharger le modèle à chaque interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model() -> MatchPredictor | None:
    if not config.MODEL_FILE.exists():
        return None
    return MatchPredictor.load()


@st.cache_data(show_spinner=False)
def get_fixtures() -> pd.DataFrame:
    return load_fixtures()


def proba_bar(probs: dict[str, float]) -> None:
    """Barre 1/N/2 sous forme de 3 colonnes de métriques."""
    c1, cn, c2 = st.columns(3)
    c1.metric("1 — Domicile", f"{probs['1']*100:.0f}%")
    cn.metric("N — Nul", f"{probs['N']*100:.0f}%")
    c2.metric("2 — Extérieur", f"{probs['2']*100:.0f}%")


# ---------------------------------------------------------------------------
# Barre latérale — paramètres de bankroll / stratégie
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Paramètres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=10.0, value=float(config.BANKROLL),
                                   step=10.0)
min_value_pct = st.sidebar.slider("Seuil de value minimal (%)", 0, 30,
                                  int(config.MIN_VALUE * 100), 1)
st.sidebar.caption(
    f"Kelly fractionné : {config.KELLY_FRACTION:.0%} du Kelly plein, "
    f"plafonné à {config.MAX_BET_FRACTION:.0%} de la bankroll."
)
st.sidebar.divider()
st.sidebar.info(
    "Mise responsable : ne pariez que de l'argent que vous pouvez perdre. "
    "Aucun modèle ne garantit de gain."
)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.title("⚽ Coupe du Monde — Détecteur de Value Bets (ML)")
st.caption("Random Forest / XGBoost calibré · stratégie de value betting · gestion Kelly")

model = get_model()
if model is None:
    st.error("Aucun modèle entraîné. Lancez d'abord : `python train.py`")
    st.stop()

meta = model.metadata
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Moteur", meta.get("backend", "?"))
mc2.metric("Matchs d'entraînement", f"{meta.get('n_train', 0):,}".replace(",", " "))
mc3.metric("Calibré", "Oui" if meta.get("calibrated") else "Non")
mc4.metric("Données jusqu'au", meta.get("last_match_date", "?"))

fixtures = get_fixtures()
if fixtures.empty:
    st.warning(f"Aucun match dans `{config.FIXTURES_CSV}`. Ajoutez-y des matchs et leurs cotes.")
    st.stop()

# ---------------------------------------------------------------------------
# Calcul des prédictions + value bets
# ---------------------------------------------------------------------------
all_value_bets: list[dict] = []
match_blocks: list[dict] = []
for r in fixtures.itertuples(index=False):
    probs = model.predict_match(r.home_team, r.away_team, r.date, bool(r.neutral), r.tournament)
    odds = odds_dict(r)
    bets = find_value_bets(probs, odds, bankroll=bankroll, min_value=min_value_pct / 100.0)
    match_blocks.append({"row": r, "probs": probs, "odds": odds, "bets": bets})
    for b in bets:
        all_value_bets.append({
            "Match": f"{r.home_team} vs {r.away_team}",
            "Date": pd.Timestamp(r.date).strftime("%d/%m"),
            "Pari": b.label,
            "Cote": b.odds,
            "Proba modèle": f"{b.model_prob*100:.0f}%",
            "Proba marché": f"{b.market_prob*100:.0f}%",
            "Value": f"+{b.edge_pct:.1f}%",
            "Mise (€)": b.stake,
        })

# ---------------------------------------------------------------------------
# Onglet 1 : alertes Value Bets (mises en évidence)
# ---------------------------------------------------------------------------
tab_value, tab_matches = st.tabs(["🔥 Value Bets", "📋 Tous les matchs"])

with tab_value:
    if all_value_bets:
        total_stake = sum(b["Mise (€)"] for b in all_value_bets)
        st.success(f"**{len(all_value_bets)} value bet(s)** détecté(s) · "
                   f"total engagé recommandé : **{total_stake:.2f} €** "
                   f"({total_stake/bankroll*100:.1f}% de la bankroll)")
        st.dataframe(pd.DataFrame(all_value_bets), use_container_width=True, hide_index=True)
        st.caption("Une value existe quand la proba du modèle dépasse la proba implicite "
                   "(1/cote, marge retirée). La mise suit le critère de Kelly fractionné.")
    else:
        st.info(f"Aucun value bet ≥ {min_value_pct}% avec les cotes actuelles. "
                "C'est normal : la majorité des matchs n'offrent pas de value.")

# ---------------------------------------------------------------------------
# Onglet 2 : détail de tous les matchs
# ---------------------------------------------------------------------------
with tab_matches:
    for block in match_blocks:
        r = block["row"]
        when = pd.Timestamp(r.date).strftime("%d/%m/%Y")
        flag = "🔥 " if block["bets"] else ""
        with st.container(border=True):
            st.subheader(f"{flag}{r.home_team} vs {r.away_team}")
            st.caption(f"{when} · {r.tournament} · "
                       f"{'terrain neutre' if bool(r.neutral) else 'à domicile'}")
            proba_bar(block["probs"])
            if block["odds"]:
                oc1, oc2, oc3 = st.columns(3)
                oc1.write(f"Cote 1 : **{block['odds'].get('1', '—')}**")
                oc2.write(f"Cote N : **{block['odds'].get('N', '—')}**")
                oc3.write(f"Cote 2 : **{block['odds'].get('2', '—')}**")
            for b in block["bets"]:
                st.success(f"💡 **Value bet — {b.label}** @ {b.odds} · "
                           f"value **+{b.edge_pct:.1f}%** · mise conseillée **{b.stake:.2f} €** "
                           f"(modèle {b.model_prob*100:.0f}% vs marché {b.market_prob*100:.0f}%)")

st.divider()
st.caption(
    f"Généré le {datetime.now(timezone.utc):%d/%m/%Y %H:%M UTC}. "
    "⚠️ Les paris sportifs comportent un risque de perte financière. "
    "Suivez votre ROI réel et arrêtez si le modèle perd de l'argent sur ≥ 30 paris."
)
