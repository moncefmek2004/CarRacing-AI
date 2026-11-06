# CarRacing AI Challenge : conduite autonome par apprentissage par renforcement

Projet individuel pour le **CarRacing AI Challenge** : entraîner un réseau de neurones qui conduit une voiture dans l'environnement [Gymnasium `CarRacing-v3`](https://gymnasium.farama.org/environments/box2d/car_racing/), en ne voyant que l'image du jeu (96×96 pixels), et qui doit boucler un tour le plus vite possible.

Le détail du parcours (essais, bugs, décisions) est dans [`journal.md`](journal.md).

## 1. Le défi en bref

| Élément       | Règle                                                                            |
|---------------|----------------------------------------------------------------------------------|
| Environnement | `CarRacing-v3`, actions continues, 3000 pas maximum (60 s)                       |
| Observation   | image RGB `uint8` de forme `(96, 96, 3)   `                                      |
| Action        | `[direction, gaz, frein]` (direction dans `[-1, 1]`, gaz et frein dans `[0, 1]`) |
| Pistes        | 3 pistes connues (seeds 202, 42, 203) et 1 piste mystère                         |
| Score         | temps au tour moyen sur 3 runs par piste ; un échec (DNF) compte 70 s            | 
| Contraintes   | l'agent final est un réseau de neurones et n'utilise que l'image                 |
| Entretien     | questions orales sur les réseaux de neurones, le RL et mon propre travail        |

## 2. Approche

- **Algorithme** : PPO (*Proximal Policy Optimization*) avec Stable-Baselines3 et PyTorch.
- **Entraînement sur CPU** (portable sans GPU, WSL Ubuntu), avec 8 environnements en parallèle.
- **Pistes aléatoires** à chaque épisode, pour que l'agent apprenne à conduire et pas à mémoriser les pistes connues. Les seeds 202, 42 et 203 servent de test.

### Prétraitement (identique à l'entraînement et à l'inférence)

1. Niveaux de gris : `0.299·R + 0.587·G + 0.114·B`.
2. *Frame skip* de 2 : une décision toutes les 2 images du jeu, l'action est répétée.
3. *Frame stack* de 4 : le réseau reçoit les 4 dernières images décisionnelles, ce qui lui donne la vitesse et le sens du mouvement.

Le tout est inclus dans la classe `DriverV4` (`driver_ppo_v4.py`) : le modèle exporté prend l'image brute et fait lui-même le prétraitement.

### Réseau

CNN de type « Nature DQN » fourni par `CnnPolicy` de Stable-Baselines3, avec une entrée de 4 canaux :

| Couche                        | Détail                                                | Sortie           |
|-------------------------------|-------------------------------------------------------|------------------|
| Entrée                        | 4 images en niveaux de gris, valeurs divisées par 255 | 4×96×96          |
| Conv 1                        | 32 filtres 8×8, pas 4, ReLU                           | 32×23×23         |
| Conv 2                        | 64 filtres 4×4, pas 2, ReLU                           | 64×10×10         |
| Conv 3                        | 64 filtres 3×3, pas 1, ReLU                           | 64×8×8           | 
| Aplatissement et couche dense | 4096 → 512, ReLU                                      | 512              |
| Sortie politique              | couche linéaire puis `tanh`                           | direction et gaz |

L'architecture exacte se vérifie avec :

```bash
python -c "from stable_baselines3 import PPO; print(PPO.load('models/BEST_v4_300k.zip').policy)"
```

### Actions (version v4)

- Le réseau sort deux valeurs bornées par `tanh` : direction et gaz.
- Le gaz est ramené à `[0, MAX_GAS]` avec `MAX_GAS = 0.5` ; le frein est fixé à 0.
- Cette bornation vient d'un problème rencontré : sans elle, les actions saturaient à leurs bornes et l'apprentissage s'arrêtait (voir le journal).

### Hyperparamètres PPO (v4)

`learning_rate` linéaire de 1e-4 à 0, `n_steps` 512 par environnement, `batch_size` 128, `n_epochs` 5, `gamma` 0,99, `gae_lambda` 0,95, `clip_range` 0,2, `target_kl` 0,03, exploration gSDE (`use_sde=True`, `log_std_init=-1`).

### Reward shaping (entraînement uniquement)

- Petite pénalité quand la voiture est sur l'herbe (détectée sur l'image).
- Arrêt de l'épisode après un long séjour sur l'herbe.
- Version v5 : arrêt aussi si la voiture ne gagne aucune tuile de piste pendant 100 décisions, avec une pénalité.

## 3. Résultats

Meilleur modèle actuel : `models/BEST_v4_300k.pt` (checkpoint de la v4 à 300 000 étapes). Évaluation déterministe, un run par piste :

| Piste            | Résultat                         |
|------------------|----------------------------------|
| 202 (piste 1)    | **16,80 s**                      |
| 42 (piste 2)     | DNF (98,9 % des tuiles visitées) |
| 203 (piste 3)    | DNF (98,3 % des tuiles visitées) |
| seeds 1, 3, 4, 6 | tour terminé (38 à 46 s)         |
| seeds 2, 5, 7    | DNF                              |

Sur 10 pistes de test, le modèle en termine 5. Tous les autres checkpoints de la v4 (50k à 500k étapes) font 10 DNF sur 10 : l'entraînement est instable, et le checkpoint de 300k a été choisi par comparaison sur plusieurs pistes.

**Observation importante** : sur plusieurs pistes, la voiture visite 94 à 99 % des tuiles puis tourne jusqu'à la fin des 3000 pas sans que le tour soit validé. D'après le code de Gymnasium, le tour n'est validé que lorsque presque toutes les tuiles sont visitées, et il en manque quelques-unes. La suite du travail porte sur cette cause.

## 4. Structure du dépôt

| Fichier | Rôle                                                                                                                        |
|-----------------------------------------------------------------|---------------------------------------------------------------------|
| `01_explore_environment.py`                                     | exploration de l'environnement avec des actions aléatoires |
| `02` à `05`                                                     | premier PPO (100k étapes), évaluation et diagnostic |
| `06` à `08`                                                     | PPO v2 (50k étapes) |
| `09_train_ppo_v3.py`, `driver_ppo.py`, `10_export_ppo_v3.py`    | v3 : 8 environnements, gris, frame stack, export `.pt` |
| `09_train_ppo_v4.py`, `driver_ppo_v4.py`, `10_export_ppo_v4.py` | v4 : actions bornées par `tanh`, gaz plafonné |
| `11_evaluate_ppo_v3.py`                                         | évaluation façon script officiel (temps au tour, DNF) |
| `12` à `16`, `18`, `21`, `22`                                   | inspection des actions, parité d'export, comparaison de checkpoints, tests visuels, diagnostic des tuiles manquantes |
| `19_train_ppo_v5.py`                                            | v5 : reprise depuis le meilleur modèle, pistes de focus et aléatoires |
| `journal.md`                                                    | journal de développement |
| `models/BEST_v4_300k.pt`                                        | modèle soumis à ce jour |

## 5. Installation et utilisation

Testé sous WSL Ubuntu 24.04, Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "gymnasium[box2d]" stable-baselines3 tensorboard
```

Entraîner (la v4, 500 000 étapes, environ 1 h 30 sur CPU) :

```bash
python 09_train_ppo_v4.py 500000
```

Exporter un checkpoint en modèle PyTorch autonome, puis l'évaluer :

```bash
python 10_export_ppo_v4.py models/checkpoints/carracing_ppo_v4_300000_steps.zip models/mon_modele.pt
RUNS=1 python 11_evaluate_ppo_v3.py models/mon_modele.pt 202 42 203
```

Regarder la voiture conduire :

```bash
python 18_test_ppo_v4.py models/BEST_v4_300k.pt 202
```

Suivre les courbes d'entraînement :

```bash
tensorboard --logdir logs
```

### Utiliser le modèle `.pt`

```python
import torch
import driver_ppo_v4  # nécessaire pour recharger la classe DriverV4

model = torch.load("models/BEST_v4_300k.pt", weights_only=False).eval()
action = model(obs)  # obs : image (96, 96, 3) uint8 ; renvoie [direction, gaz, frein]
```

Le modèle garde un état interne (les dernières images) : il faut le **recharger au début de chaque épisode**, ou appeler `model.reset()`.

## 6. Ce que j'ai appris

- Une action coupée à ses bornes ne transmet plus de signal d'apprentissage : le bruit d'exploration n'a plus d'effet.
- La politique stochastique de l'entraînement et la politique déterministe de l'évaluation peuvent se comporter très différemment ; `ep_rew_mean` pendant l'entraînement n'était pas un bon indicateur de la conduite réelle.
- Il faut comparer les checkpoints sur plusieurs pistes et garder le meilleur, pas le dernier.
- Vérifier l'export : le modèle exporté donne exactement les mêmes actions que celui de Stable-Baselines3 (écart de 0,0).
- Lire le code de l'environnement : la règle réelle de fin de tour diffère un peu de celle annoncée.

## 7. Pistes d'amélioration

- Comprendre quelles tuiles sont manquées et pourquoi (virages coupés ?), puis ajouter un bonus de fin de tour.
- Relever le plafond de gaz une fois la conduite fiable, pour gagner du temps au tour.
- Entraîner plus longtemps, avec des checkpoints plus fréquents et une sélection sur davantage de pistes.