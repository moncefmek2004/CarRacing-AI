import sys

import gymnasium as gym
import numpy as np
import torch

import driver_ppo  # noqa: F401
import driver_ppo_v4  # noqa: F401

PATH = sys.argv[1] if len(sys.argv) > 1 else "models/carracing_ppo_v3_driver.pt"
env = gym.make("CarRacing-v3", max_episode_steps=3000)
model = torch.load(PATH, weights_only=False).eval()
obs, _ = env.reset(seed=202)

acts, total = [], 0.0
for step in range(1, 1001):
    a = model(obs)
    acts.append(a)
    obs, r, terminated, truncated, _ = env.step(a)
    total += r
    if step % 100 == 0:
        print(f"etape {step:4d} | action {np.round(a, 2)} | reward cumulee {total:.1f}")
    if terminated or truncated:
        break

acts = np.array(acts)
for name, v in [("moyenne", acts.mean(0)), ("ecart-type", acts.std(0)),
                ("min", acts.min(0)), ("max", acts.max(0))]:
    print(f"{name:10s} [direction, gaz, frein] = {np.round(v, 3)}")
