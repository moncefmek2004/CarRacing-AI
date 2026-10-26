import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/carracing_ppo_v2_50k.zip"
SEED = 202
TOTAL_STEPS = 3000

env = gym.make("CarRacing-v3", max_episode_steps=TOTAL_STEPS)

try:
    model = PPO.load(MODEL_PATH, device="auto")
    observation, info = env.reset(seed=SEED)

    total_reward = 0.0
    actions = []

    print("Evaluation du modele PPO v2...")
    print("Modele :", MODEL_PATH)

    for step in range(1, TOTAL_STEPS + 1):
        action, _ = model.predict(observation, deterministic=True)
        actions.append(action.copy())

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 500 == 0:
            print(
                f"Etape {step} | "
                f"recompense cumulee : {total_reward:.2f} | "
                f"direction : {action[0]:+.3f} | "
                f"acceleration : {action[1]:.3f} | "
                f"frein : {action[2]:.3f}"
            )

        if terminated or truncated:
            print(f"Episode termine a l'etape {step}.")
            break

    actions = __import__("numpy").asarray(actions)

    print("\n=== RESULTAT FINAL ===")
    print(f"Recompense totale : {total_reward:.2f}")
    print(f"Direction moyenne : {actions[:, 0].mean():+.3f}")
    print(f"Acceleration moyenne : {actions[:, 1].mean():.3f}")
    print(f"Frein moyen : {actions[:, 2].mean():.3f}")

finally:
    env.close()
