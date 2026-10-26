
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/carracing_ppo_100k"

env = gym.make(
    "CarRacing-v3",
    render_mode="human",
    max_episode_steps=3000,
)

try:
    model = PPO.load(MODEL_PATH)

    observation, info = env.reset(seed=202)
    total_reward = 0.0

    print("Evaluation du modele PPO entraine sur 100 000 etapes")

    for step in range(3000):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 500 == 0:
            print(
                f"Etape : {step + 1} | "
                f"Recompense totale : {total_reward:.2f}"
            )

        if terminated or truncated:
            print("Episode termine.")
            break

    print(f"\nRecompense finale : {total_reward:.2f}")

finally:
    env.close()
