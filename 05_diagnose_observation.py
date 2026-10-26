from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

MODEL_PATH = "models/carracing_ppo_100k"
OUTPUT_DIR = Path("diagnostic_images")
TOTAL_STEPS = 300

OUTPUT_DIR.mkdir(exist_ok=True)

env = gym.make("CarRacing-v3", max_episode_steps=3000)

try:
    print("Chargement du modele PPO...")
    model = PPO.load(MODEL_PATH, device="auto")

    observation, info = env.reset(seed=202)

    print("\n--- Diagnostic de l'observation ---")
    print(f"Forme : {observation.shape}")
    print(f"Type : {observation.dtype}")
    print(f"Valeur minimale : {observation.min()}")
    print(f"Valeur maximale : {observation.max()}")
    print(f"Moyenne des pixels : {observation.mean():.2f}")

    Image.fromarray(observation).save(
        OUTPUT_DIR / "observation_initiale.png"
    )
    print("Image initiale sauvegardee.")

    actions = []
    rewards = []
    terminated = False
    truncated = False

    for step in range(1, TOTAL_STEPS + 1):
        action, _ = model.predict(observation, deterministic=True)

        actions.append(action.copy())

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if step in (50, 100, 200, 300):
            Image.fromarray(observation).save(
                OUTPUT_DIR / f"observation_step_{step}.png"
            )

        if step % 25 == 0:
            print(
                f"Etape {step:3d} | "
                f"direction={action[0]:+.3f} | "
                f"acceleration={action[1]:.3f} | "
                f"frein={action[2]:.3f} | "
                f"recompense={reward:+.3f}"
            )

        if terminated or truncated:
            print(f"\nEpisode termine a l'etape {step}.")
            break

    actions = np.asarray(actions)
    rewards = np.asarray(rewards)

    print("\n--- Statistiques des actions ---")
    print(f"Nombre d'actions : {len(actions)}")
    print(f"Direction moyenne : {actions[:, 0].mean():+.3f}")
    print(f"Direction min/max : {actions[:, 0].min():+.3f} / {actions[:, 0].max():+.3f}")
    print(f"Acceleration moyenne : {actions[:, 1].mean():.3f}")
    print(f"Frein moyen : {actions[:, 2].mean():.3f}")
    print(f"Recompense totale : {rewards.sum():.2f}")

    print(f"\nImages sauvegardees dans : {OUTPUT_DIR.resolve()}")
    print("Diagnostic termine.")

finally:
    env.close()
