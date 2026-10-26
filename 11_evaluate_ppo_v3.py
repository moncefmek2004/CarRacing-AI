"""Évalue un .pt exporté comme le script officiel : 3 runs par piste,
DNF = 70 s. Temps du tour = env.unwrapped.t si lap_finished.

Usage : python 11_evaluate_ppo_v3.py [modele.pt] [seed1 seed2 ...]
"""
import sys

import gymnasium as gym
import numpy as np
import torch

import driver_ppo  # noqa: F401  (nécessaire pour recharger la classe Driver)
import driver_ppo_v4  # noqa: F401

PATH = sys.argv[1] if len(sys.argv) > 1 else "models/carracing_ppo_v3_driver.pt"
SEEDS = [int(s) for s in sys.argv[2:]] or [202, 42, 203]
import os
RUNS = int(os.environ.get("RUNS", "3"))
DNF_TIME = 70.0


def run(seed):
    env = gym.make("CarRacing-v3", max_episode_steps=3000)
    model = torch.load(PATH, weights_only=False)
    model.eval()
    obs, _ = env.reset(seed=seed)
    done, info, reward_sum = False, {}, 0.0
    while not done:
        action = model(obs)
        obs, r, terminated, truncated, info = env.step(action)
        reward_sum += r
        done = terminated or truncated
    lap = env.unwrapped.t if info.get("lap_finished", False) else None
    env.close()
    return lap, reward_sum


for seed in SEEDS:
    times = []
    for i in range(RUNS):
        lap, rew = run(seed)
        times.append(lap if lap is not None else DNF_TIME)
        print(f"seed {seed} run {i+1}: "
              f"{'DNF' if lap is None else f'{lap:.2f}s'} | reward {rew:.1f}")
    print(f"==> seed {seed}: moyenne {np.mean(times):.2f}s "
          f"({sum(t == DNF_TIME for t in times)} DNF)\n")
