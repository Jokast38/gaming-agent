# Gaming Agent — Snake + Q-Learning / DQN

## Équipe
- Nom de team : Snake gaming agent
- Membres : Jokast; Serge; Herrmann; Rufus

## Le jeu
Snake, implémenté à la main avec PyGame (grille 20x15 cases). Choisi car l'environnement
tourne et donne un score en quelques minutes de mise en place, sans dépendance lourde
(pas de GPU nécessaire), et permet un espace d'états discrétisable simplement pour du
Q-learning tabulaire.

## Ce que l'agent observe, fait, et ce qui le récompense

**Observation (état, 15 booléens)** :
- 3 dangers immédiats à 1 case (tout droit / à droite / à gauche du sens de déplacement)
- 3 dangers anticipés à 2 cases (mêmes directions, pour voir un peu plus loin)
- direction actuelle du serpent (haut/bas/gauche/droite)
- position relative de la nourriture (gauche/droite/haut/bas par rapport à la tête)
- le serpent est-il "long" (>15 cases), pour distinguer le comportement prudent en fin de
  partie du comportement en début de partie

Cela donne un espace d'états discret (jusqu'à 2^15 = 32768 combinaisons), toujours gérable
par une table Q, et réutilisable tel quel comme entrée d'un petit réseau de neurones (DQN).

**Actions (3, relatives à la direction courante)** : tout droit, tourner à droite, tourner à gauche.

**Récompense** :
- +10 quand le serpent mange la nourriture
- -10 quand la partie se termine (collision avec un mur ou avec lui-même, ou trop de pas
  sans manger)
- +1 / -1 à chaque pas selon que le serpent se rapproche ou s'éloigne de la nourriture
  (reward shaping, ajouté après coup pour aider l'agent tabulaire à naviguer — voir
  [NOTEBOOK.md](NOTEBOOK.md))

Fonction volontairement simple au départ (+10/-10 seulement), complexifiée une fois que le
temps le permettait, conformément à la consigne du sujet.

## Méthodes d'apprentissage

**1) Q-learning tabulaire** (choix initial). Choisi parce que l'état est discrétisé en un
nombre raisonnable de combinaisons, ce qui rend une table Q (dictionnaire état→valeurs
d'actions) suffisante, sans avoir besoin d'un réseau de neurones. C'est aussi la méthode la
plus simple à faire tourner et à comprendre entièrement en 2 jours, sans GPU.

Hyperparamètres (`agent/q_learning.py`) : learning rate = 0.1, gamma = 0.9, epsilon initial
= 1.0, décroissance ×0.995 (ou ×0.999 pour les runs longs) par épisode, minimum 0.01.

**2) DQN (Deep Q-Network)** (amélioration testée ensuite). Une fois le plafond du tabulaire
observé (~27 de score moyen malgré plus d'épisodes et de features), on a testé un DQN — même
état en entrée, mais approximé par un petit réseau de neurones (2 couches cachées de 256
neurones) plutôt qu'une table, avec replay buffer et réseau cible. L'idée : généraliser sur
des états jamais vus exactement pendant l'entraînement, ce qu'une table ne sait pas faire.

Hyperparamètres (`agent/dqn_agent.py`) : learning rate = 1e-3 (Adam), gamma = 0.9, epsilon
initial = 1.0, décroissance ×0.999, batch size = 64, réseau cible mis à jour toutes les 200
étapes.

## Résultats

| Agent | Score moyen | Max | Min | Nb parties |
|---|---|---|---|---|
| Aléatoire (référence) | 0.20 | 1 | 0 | 20 |
| Q-learning tabulaire, état simple (2000 épisodes) | 19.00 | 38 | 9 | 20 |
| Q-learning tabulaire, état enrichi + reward shaping (8000 épisodes) | 26.95 | 43 | 12 | 20 |
| DQN (5000 épisodes) | 30.45 | 57 | 10 | 20 |

L'agent DQN final fait ~150x mieux que le hasard sur le même nombre de parties, et environ
13% mieux que le meilleur agent tabulaire. Le score maximal théorique de la grille (297,
remplir les 300 cases) reste hors de portée : au-delà d'un certain score le serpent devient
trop long pour que ces représentations d'état (locales, sans vue globale de la grille)
permettent d'anticiper un auto-enfermement — c'est documenté comme plafond structurel dans
le carnet d'essais.

Courbe de progression : voir `runs/<nom_run>/learning_curve.png` (générée localement,
non versionnée — voir [NOTEBOOK.md](NOTEBOOK.md) pour l'historique complet des essais, y
compris les tentatives et leurs résultats détaillés).

## Comment lancer

```bash
pip install -r requirements.txt

# Agent aléatoire jouable (référence)
python game/play_random.py --episodes 20 --render

# --- Q-learning tabulaire ---
python train.py --episodes 2000 --run-name essai1
python evaluate.py --model runs/essai1/best_agent.pkl --episodes 20 --render

# --- DQN ---
python train_dqn.py --episodes 5000 --run-name dqn_essai1 --epsilon-decay 0.999
python evaluate_dqn.py --model runs/dqn_essai1/best_agent.pt --episodes 20 --render
```

## Carnet d'essais
Voir [NOTEBOOK.md](NOTEBOOK.md) pour l'historique des tentatives, y compris les échecs.

## Vidéo de présentation
Lien : https://testipformation-my.sharepoint.com/:v:/g/personal/s_donou_ecole-ipssi_net/IQDcwU1jTYMFS7of8b5v7yQEAVIB-t2QrNEGrxWCxX6oIZY?e=k2bmaj

## Ce qu'on ferait avec plus de temps
Faire un agent capable de jouer les échecs sur la plateforme: https://papergames.io/fr/jeu-d-echecs