from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

env = gym.make("CarRacing-v3", max_episode_steps=3000)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path="./models/checkpoints/",
    name_prefix="carracing_ppo_v2",
)

try:
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        verbose=1,
        seed=202,
        device="auto",
        tensorboard_log="./logs/",
    )

    print("Debut de l'experience PPO v2 : 50 000 etapes")

    model.learn(
        total_timesteps=50_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_CarRacing_v2",
    )

    model.save("models/carracing_ppo_v2_50k")

    print("\nExperience terminee.")
    print("Modele : models/carracing_ppo_v2_50k.zip")
    print("Checkpoints : models/checkpoints/")

finally:
    env.close()
