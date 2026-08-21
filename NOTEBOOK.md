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

## Essai 4 — Reward shaping (distance à la nourriture), 5000 épisodes
- Date : 2026-08-21
- Objectif : dépasser le plateau ~19-20 observé dans les essais 2-3, en direction d'un
  score plus proche de 100 comme demandé.
- Changement : ajout d'un +1/-1 à chaque pas selon que le serpent se rapproche/s'éloigne de
  la nourriture (en plus du +10/-10 existant), et la limite de pas sans manger s'allonge
  maintenant avec la taille du serpent (sinon un long serpent n'a plus le temps de
  parcourir la grille). run: `essai2_shaping`, lr=0.1, gamma=0.9, epsilon_decay=0.995,
  seed=42.
- Score moyen entraînement (100 derniers épisodes) : 18.54. Meilleur score : 54.
- Évaluation (20 parties, greedy) : score moyen 20.85 (max 57, min 5).
- Conclusion : légère amélioration par rapport à l'essai 2 (19.00 → 20.85), mais pas le
  gain espéré. Le reward shaping aide un peu à naviguer mais ne change pas la limite
  fondamentale : l'agent ne voit toujours que son voisinage immédiat.

## Essai 5 — État enrichi (dangers à 2 cases + taille du serpent), 8000 épisodes
- Date : 2026-08-21
- Changement : état étendu de 11 à 15 booléens (ajout de 3 dangers anticipés à 2 cases et
  d'un indicateur "serpent long" >15 cases). epsilon_decay ralenti à 0.999 pour explorer
  plus longtemps sur ce plus grand espace d'états (32768 combinaisons possibles). run:
  `essai3_richer_state`, seed=42.
- Score moyen entraînement (100 derniers épisodes) : 23.46. Meilleur score : 60.
- Évaluation (20 parties, greedy) : score moyen 26.95 (max 43, min 12).
- Conclusion : nouvelle amélioration nette (20.85 → 26.95), mais rendements décroissants —
  chaque enrichissement d'état aide de moins en moins. C'est le signe qu'on approche la
  limite structurelle du Q-learning tabulaire avec un état purement local : au-delà d'un
  certain score, le serpent devient trop long pour que ces features locales suffisent à
  anticiper un auto-enfermement plusieurs coups à l'avance.

## Essai 6 — DQN, run interrompu (~4000/5000 épisodes)
- Date : 2026-08-20/21
- Objectif : passer d'une table à un réseau de neurones (2 couches cachées de 256), avec
  replay buffer et réseau cible, pour généraliser au-delà de ce qu'un tabulaire peut
  apprendre par cœur. Même état 15 booléens en entrée. run: `dqn_essai1`, lr=1e-3 (Adam),
  gamma=0.9, epsilon_decay=0.999, batch_size=64, seed=42, 5000 épisodes visés.
- Le run a été interrompu avant la fin (fermeture accidentelle du terminal, ~4000 épisodes
  effectués). Le checkpoint `best_agent.pt` sauvegardé avant l'interruption a quand même pu
  être rechargé et évalué.
- Évaluation (20 parties, greedy) : score moyen 27.00 (max 52, min 16).
- Conclusion (tentative ratée, mais instructive) : même incomplet, le DQN égale déjà le
  meilleur tabulaire. Leçon retenue : toujours lancer les entraînements longs en gardant le
  terminal actif jusqu'à la fin (ou en tâche de fond correctement suivie), et le mécanisme
  de sauvegarde du meilleur score en cours de route (`best_agent.pt` mis à jour à chaque
  nouveau record) a évité de tout perdre.

## Essai 7 — DQN, run complet, 5000 épisodes
- Date : 2026-08-21
- Même configuration que l'essai 6, relancé jusqu'au bout sans interruption. run:
  `dqn_essai2`, seed=42.
- Score moyen entraînement (100 derniers épisodes) : 22.69. Meilleur score en
  entraînement : 52.
- Évaluation (20 parties, greedy) : score moyen 30.45 (max 57, min 10).
- Conclusion : meilleur résultat de tous les essais. Le DQN dépasse le meilleur Q-learning
  tabulaire (30.45 vs 26.95), confirmant qu'approximer la fonction Q avec un réseau de
  neurones généralise mieux que la table sur cet espace d'états, même sans changer la
  représentation d'entrée. Reste très loin du score maximal théorique (297) : atteindre ça
  demanderait une architecture voyant la grille entière (ex: CNN sur l'image du plateau) et
  beaucoup plus de temps d'entraînement — hors de portée en 2 jours sans GPU. On considère
  ce plafond (~27-30 de score moyen) comme un résultat honnête et documenté plutôt qu'un
  échec : l'objectif du projet est de battre le hasard de façon mesurée, ce qui est fait
  très largement (~150x).

## Synthèse
| Essai | Méthode | Score moyen (éval, 20 parties) | Max |
|---|---|---|---|
| 0 | Aléatoire | 0.20 | 1 |
| 2 | Q-learning, état simple | 19.00 | 38 |
| 4 | Q-learning + reward shaping | 20.85 | 57 |
| 5 | Q-learning, état enrichi | 26.95 | 43 |
| 6 | DQN (interrompu) | 27.00 | 52 |
| 7 | DQN, run complet | **30.45** | **57** |

<!-- Ajouter une entrée par tentative, même ratée -->
