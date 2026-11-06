import sys

import gymnasium as gym
import torch

import driver_ppo_v4  # noqa: F401

path = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]] or [202, 42, 203, 43]
for seed in seeds:
    model = torch.load(path, weights_only=False).eval()
    env = gym.make("CarRacing-v3", max_episode_steps=3000)
    obs, _ = env.reset(seed=seed)
    total, last_r, info = 0.0, 0.0, {}
    for step in range(1, 3001):
        obs, r, term, trunc, info = env.step(model(obs))
        total += r
        last_r = r
        if term or trunc:
            break
    u = env.unwrapped
    ratio = u.tile_visited_count / len(u.track)
    if info.get("lap_finished"):
        reason = "tour fini"
    elif last_r <= -99:
        reason = "hors terrain"
    else:
        reason = "temps ecoule"
    print(f"seed {seed}: {reason} | pas {step} | tuiles {ratio:.1%} | reward {total:.0f}")
    env.close()
