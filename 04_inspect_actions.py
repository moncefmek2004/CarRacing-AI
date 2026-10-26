
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "models/carracing_ppo_initial"

env = gym.make("CarRacing-v3", max_episode_steps=3000)

try:
    model = PPO.load(MODEL_PATH)

    observation, info = env.reset(seed=202)

    actions = []
    total_reward = 0.0

    print("Analyse du modele PPO")
    print("Format : [direction, acceleration, frein]\n")

    for step in range(100):
        # Le modele choisit une action
        action, _ = model.predict(observation, deterministic=True)

        # Afficher l'action choisie
        print(f"Etape {step + 1:3d} | Action : {action}")

        actions.append(action)

        # Executer l'action dans le jeu
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Afficher la recompense toutes les 10 etapes
        if (step + 1) % 10 == 0:
            print(
                f"Etape {step + 1:3d} | "
                f"Recompense : {reward:.3f} | "
                f"Total : {total_reward:.3f}"
            )

        if terminated or truncated:
            print("La partie est terminee.")
            break

    # Calculer les statistiques des actions
    actions = np.array(actions)

    print("\n--- Statistiques des actions ---")
    print("Nombre d'actions :", len(actions))
    print("Moyenne direction :", round(float(actions[:, 0].mean()), 3))
    print("Moyenne acceleration :", round(float(actions[:, 1].mean()), 3))
    print("Moyenne freinage :", round(float(actions[:, 2].mean()), 3))
    print("Variation direction :", round(float(actions[:, 0].std()), 3))
    print("Recompense totale :", round(float(total_reward), 3))

finally:
    env.close()
