import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

import driver_ppo_v4  # noqa: F401

zip_path = sys.argv[1]
pt_path = sys.argv[2]
sb3 = PPO.load(zip_path, device="cpu")
drv = torch.load(pt_path, weights_only=False).eval()
env = gym.make("CarRacing-v3", max_episode_steps=3000)
obs, _ = env.reset(seed=202)

diffs = []
for step in range(300):
    a = drv(obs)
    if step % drv.skip == 0:
        stack = (drv.stack.numpy() * 255).round().astype(np.uint8)
        ref, _ = sb3.predict(stack, deterministic=True)
        expected = np.array([ref[0], (ref[1] + 1) / 2 * drv.max_gas, 0.0])
        diffs.append(np.abs(expected - a).max())
    obs, *_ = env.step(a)
print("ecart max :", max(diffs))
