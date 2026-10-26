# Journal de développement : CarRacing AI Challenge

## Approche
PPO (Stable-Baselines3, CnnPolicy) sur CarRacing-v3, entraînement sur CPU (WSL, portable sans GPU).

## Chronologie

### Étape 1 : PPO de base (100k étapes)
Image RGB brute, un seul environnement, sans frame stack. Reward sur la seed 202 : -281,88. La voiture ne conduit pas.

### Étape 2 : PPO v2 (50k étapes)
Learning rate plus bas, clip 0,1. Reward : -267,39. Écart de +14 points (environ 4 tuiles), sans signification : -0,1 par pas sans progresser donne -300 sur 3000 pas.

### Étape 3 : PPO v3 (100k étapes)
Changements : 8 environnements en parallèle, niveaux de gris, frame skip 2, frame stack 4, pistes aléatoires, pénalité d'herbe, arrêt anticipé hors piste.
- Bug : `ep_rew_mean` absent du log. Cause : le Monitor était placé avant mon wrapper, il ne voyait pas les épisodes arrêtés par l'herbe. Correction : VecMonitor.
- Problème : le modèle exporté sortait toujours l'action [1, 1, 0] (direction et gaz saturés). L'action est coupée à ses bornes, donc le bruit d'exploration ne change plus rien et l'apprentissage s'arrête.
- Vérification : mon Driver exporté donne exactement les mêmes actions que SB3 (écart 0,0). Le défaut venait donc de l'entraînement.

### Étape 4 : PPO v4 (500k étapes)
Changements : actions bornées par tanh, deux actions (direction, gaz), gaz plafonné à 0,5, frein supprimé, arrêt hors piste plus tardif et sans pénalité forte, target_kl 0,03, log_std_init -1, n_epochs 5.
- Observation : `ep_rew_mean` restait autour de -7 pendant l'entraînement, car la politique stochastique sort de la piste avec le bruit d'exploration. En mode déterministe, le checkpoint de 300k conduit.
- Résultat du meilleur modèle (checkpoint 300k) : 16,80 s sur la seed 202, DNF sur 42 et 203, 5 pistes sur 10 terminées (moyenne 53,7 s avec DNF à 70 s).
- Problème : tous les autres checkpoints (50k à 500k et final) font 10 DNF sur 10. Cause pas encore établie ; hypothèse : la voiture apprend à rester presque immobile.

## Ce que j'ai appris
- Une action coupée à ses bornes ne transmet plus de signal d'apprentissage.
- La politique stochastique (entraînement) et la politique déterministe (évaluation) peuvent avoir des performances très différentes.
- Il faut tester les checkpoints sur plusieurs pistes et garder le meilleur, pas le dernier.

## Prochaines étapes
Comprendre pourquoi les checkpoints après 300k échouent, puis reprendre l'entraînement depuis le checkpoint de 300k.
