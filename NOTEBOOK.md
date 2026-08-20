# Carnet d'essais

Format par entrée : date, changement testé, résultat chiffré, ce qu'on en conclut.
Inclure les tentatives ratées.

## Essai 0 — Référence aléatoire
- Date : 2026-08-20
- Agent aléatoire (`game/play_random.py`), 20 parties.
- Score moyen : 0.20 (max 1, min 0)
- Sert de référence pour tout le reste.

## Essai 1 — Q-learning, config par défaut, 300 épisodes (smoke test)
- Date : 2026-08-20
- lr=0.1, gamma=0.9, epsilon_decay=0.995, seed=42
- Juste un test rapide pour vérifier que le pipeline entraîne/sauvegarde/recharge
  correctement avant de lancer un run plus long.
- Score moyen (10 parties, greedy, agent rechargé) : 6.90 (max 20, min 0)
- Conclusion : le pipeline fonctionne, l'agent apprend déjà nettement mieux que le hasard
  en seulement 300 épisodes. On passe à un run plus long pour les résultats officiels.

## Essai 2 — Q-learning, config par défaut, 2000 épisodes (run officiel)
- Date : 2026-08-20
- lr=0.1, gamma=0.9, epsilon_decay=0.995, seed=42, run: `essai1`
- Moyenne mobile (50 derniers épisodes d'entraînement) : passe de ~0.2 à ~20 sur les 2000
  épisodes, se stabilise autour de 15-22 après ~800 épisodes (epsilon proche du minimum).
- Meilleur score en entraînement : 50
- Évaluation finale (best_agent.pkl rechargé depuis `evaluate.py`, 20 parties, greedy) :
  score moyen 19.00 (max 38, min 9) — comparable à l'agent aléatoire (0.20), soit ~95x
  mieux.
- Conclusion : la représentation d'état à 11 booléens (dangers + direction + position
  nourriture) suffit à un Q-learning tabulaire pour apprendre une politique largement
  meilleure que le hasard, sans réseau de neurones.

## Essai 3 — Reproductibilité, même config, seed différente
- Date : 2026-08-20
- Même hyperparamètres que l'essai 2, seed=7, run: `essai1_rerun`, 2000 épisodes.
- Moyenne mobile finale : 20.32-20.86 en fin d'entraînement (proche de l'essai 2).
- Score moyen sur 100 derniers épisodes d'entraînement : 19.18 (vs 19.56 pour l'essai 2)
- Meilleur score en entraînement : 46 (vs 50 pour l'essai 2)
- Conclusion : les deux courbes se ressemblent fortement (même ordre de grandeur, même
  forme générale de progression). Le Q-learning tabulaire sur cet espace d'états est donc
  raisonnablement stable d'un run à l'autre avec cette config — pas de signe
  d'instabilité ou de surapprentissage erratique.

<!-- Ajouter une entrée par tentative, même ratée -->
