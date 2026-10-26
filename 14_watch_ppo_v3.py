"""Regarder un checkpoint PPO conduire.

Usage : python 14_watch_ppo_v3.py <checkpoint.zip> [seed] [human|gif]
"""
import copy
import sys

import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO

from driver_ppo import Driver

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 202
mode = sys.argv[3] if len(sys.argv) > 3 else "human"

p = PPO.load(path, device="cpu").policy.eval()
drv = Driver(
    cnn=copy.deepcopy(p.features_extractor.cnn),
    linear=copy.deepcopy(p.features_extractor.linear),
    policy_net=copy.deepcopy(p.mlp_extractor.policy_net),
    action_net=copy.deepcopy(p.action_net),
).eval()

env = gym.make(
    "CarRacing-v3",
    render_mode="human" if mode == "human" else "rgb_array",
    max_episode_steps=3000,
)
obs, _ = env.reset(seed=seed)
total, frames = 0.0, []
for step in range(3000):
    obs, r, terminated, truncated, _ = env.step(drv(obs))
    total += r
    if mode == "gif" and step % 4 == 0:
        frames.append(Image.fromarray(env.render()).resize((300, 200)))
    if terminated or truncated:
        break
env.close()
print(f"{step + 1} etapes | reward {total:.1f}")

if mode == "gif":
    frames[0].save("watch.gif", save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print("GIF : watch.gif")
