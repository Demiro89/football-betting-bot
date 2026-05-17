# Football Betting Bot

Bot intelligent de prédiction et value betting pour les matchs de football (focus Ligue 1).

## Fonctionnalités actuelles
- Récupération automatique des matchs à venir via API-Football
- Calcul dynamique des lambdas (attaque/défense + forme récente)
- Modèle Poisson + Monte Carlo pour probabilités 1N2 et Over/Under

## Installation

1. Clone le repo
   ```bash
git clone https://github.com/Demiro89/football-betting-bot.git
   cd football-betting-bot
   ```

2. Installe les dépendances
   ```bash
   pip install -r requirements.txt
   ```

3. Crée un fichier `.env` avec ta clé API :
   ```env
   API_FOOTBALL_KEY=ta_cle_ici
   ```

4. Lance le bot
   ```bash
   python main.py
   ```

## Prochaines étapes
- Intégration des cotes FDJ
- Détection automatique de value bets
- Bankroll management
- Interface Telegram

Développé pour **battre la FDJ** sur le long terme.