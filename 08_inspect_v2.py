from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

output_dir = Path("diagnostic_v2")
output_dir.mkdir(exist_ok=True)

env = gym.make("CarRacing-v3", max_episode_steps=3000)

try:
    model = PPO.load("models/carracing_ppo_v2_50k.zip", device="auto")
    observation, info = env.reset(seed=202)

    total_reward = 0.0

    for step in range(1, 1001):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step in (1, 100, 250, 500, 750, 1000):
            Image.fromarray(observation).save(
                output_dir / f"observation_{step}.png"
            )
            print(
                f"Etape {step}: direction={action[0]:+.3f}, "
                f"acceleration={action[1]:.3f}, "
                f"frein={action[2]:.3f}, "
                f"recompense={total_reward:.2f}"
            )

        if terminated or truncated:
            print("Episode termine a l'etape", step)
            break

    print("\nImages sauvegardees dans :", output_dir.resolve())

finally:
    env.close()
