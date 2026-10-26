
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Creer les dossiers necessaires
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Environnement de jeu
env = gym.make("CarRacing-v3", max_episode_steps=3000)

# Sauvegarder un checkpoint toutes les 25 000 etapes
checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path="./models/checkpoints/",
    name_prefix="carracing_ppo",
)

try:
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        verbose=1,
        seed=202,
        device="auto",
        tensorboard_log="./logs/",
    )

    print("Debut de l'entrainement : 100 000 etapes")

    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_CarRacing",
    )

    # Sauvegarder le modele final separement
    model.save("models/carracing_ppo_100k")

    print("\nEntrainement termine !")
    print("Modele sauvegarde : models/carracing_ppo_100k.zip")
    print("Les checkpoints sont dans models/checkpoints/")

finally:
    env.close()
