import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

import driver_ppo  # noqa: F401

sb3 = PPO.load("models/carracing_ppo_v3_final.zip", device="cpu")
drv = torch.load("models/carracing_ppo_v3_driver.pt", weights_only=False).eval()
env = gym.make("CarRacing-v3", max_episode_steps=3000)
obs, _ = env.reset(seed=202)

diffs = []
for step in range(300):
    a = drv(obs)
    if step % drv.skip == 0:
        stack = (drv.stack.numpy() * 255).round().astype(np.uint8)
        ref, _ = sb3.predict(stack, deterministic=True)
        diffs.append(np.abs(ref - a).max())
    obs, *_ = env.step(a)
print("ecart max entre Driver et SB3 :", max(diffs))
print("derniere action SB3 :", ref, "| Driver :", a)
