import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import driver_ppo_v4  # noqa: F401


def on_grass(obs):
    p = obs[66:78, 44:52].astype(np.int16)
    green = (p[..., 1] - p[..., 0] > 40) & (p[..., 1] - p[..., 2] > 40)
    return green.mean() > 0.5


model = torch.load(sys.argv[1], weights_only=False).eval()
env = gym.make("CarRacing-v3", max_episode_steps=3000)
obs, _ = env.reset(seed=202)
Path("diag_v4").mkdir(exist_ok=True)

flags = []
for step in range(600):
    obs, r, terminated, truncated, _ = env.step(model(obs))
    flags.append(on_grass(obs))
    if step in (0, 50, 100, 200, 300, 400, 599):
        Image.fromarray(obs).save(f"diag_v4/step_{step}.png")
    if terminated or truncated:
        break

flags = np.array(flags)
first = int(np.argmax(flags)) if flags.any() else -1
print("pas jouees :", len(flags))
print("part du temps sur l'herbe :", round(float(flags.mean()), 2))
print("premier pas sur l'herbe :", first)
print("herbe aux pas 0, 30, 60, 100 :", [bool(flags[i]) for i in (0, 30, 60, 100) if i < len(flags)])
