"""Test visuel d'un .pt v4 : fenetre en direct si possible, sinon GIF.
Usage : python 18_test_ppo_v4.py modele.pt [seed1 seed2 ...]
"""
import os
import sys

import gymnasium as gym
import torch
from PIL import Image

import driver_ppo_v4  # noqa: F401

path = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]] or [202]
live = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
print("Mode :", "fenetre en direct" if live else "GIF (pas d'ecran detecte)")

for seed in seeds:
    model = torch.load(path, weights_only=False).eval()
    env = gym.make("CarRacing-v3", max_episode_steps=3000,
                   render_mode="human" if live else "rgb_array")
    obs, _ = env.reset(seed=seed)
    done, info, total, frames = False, {}, 0.0, []
    while not done:
        obs, r, terminated, truncated, info = env.step(model(obs))
        total += r
        done = terminated or truncated
        if not live and env.unwrapped.t % 4 == 0:
            frames.append(Image.fromarray(env.render()).resize((300, 200)))
    lap = env.unwrapped.t if info.get("lap_finished", False) else None
    env.close()
    verdict = f"{lap:.2f} s" if lap is not None else "DNF"
    print(f"seed {seed} : {verdict} | reward {total:.1f}")
    if not live and frames:
        out = f"test_seed{seed}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=80, loop=0)
        print("GIF :", out)
