"""Compare les checkpoints PPO v4 : temps moyen (DNF = 70 s), nombre de DNF,
cause du DNF (HORS = sorti du terrain, TEMPS = 3000 pas ecoules).

Usage : python 16_compare_ppo_v4.py [seed1 seed2 ...]
"""
import copy
import glob
import re
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from driver_ppo_v4 import DriverV4

SEEDS = [int(s) for s in sys.argv[1:]] or [202, 42, 203, 1, 2, 3, 4, 5, 6, 7]
DNF_TIME = 70.0


def load_driver(path):
    p = PPO.load(path, device="cpu").policy.eval()
    return DriverV4(
        cnn=copy.deepcopy(p.features_extractor.cnn),
        linear=copy.deepcopy(p.features_extractor.linear),
        policy_net=copy.deepcopy(p.mlp_extractor.policy_net),
        action_net=copy.deepcopy(p.action_net),
    ).eval()


def run(drv, seed):
    drv.reset()
    env = gym.make("CarRacing-v3", max_episode_steps=3000)
    obs, _ = env.reset(seed=seed)
    done, info, r = False, {}, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(drv(obs))
        done = term or trunc
    lap = env.unwrapped.t if info.get("lap_finished", False) else None
    env.close()
    if lap is not None:
        return lap, "OK"
    return DNF_TIME, ("HORS" if r <= -99 else "TEMPS")


def steps_of(path):
    m = re.search(r"_(\d+)_steps", path)
    return int(m.group(1)) if m else 10**9


paths = sorted(glob.glob("models/checkpoints/carracing_ppo_v4_*_steps.zip"),
               key=steps_of)
paths.append("models/carracing_ppo_v4_final.zip")

for path in paths:
    drv = load_driver(path)
    res = [run(drv, s) for s in SEEDS]
    times = [t for t, _ in res]
    n_dnf = sum(c != "OK" for _, c in res)
    detail = " ".join(f"{s}:{'DNF-' + c if c != 'OK' else f'{t:.0f}s'}"
                      for s, (t, c) in zip(SEEDS, res))
    print(f"{path.split('/')[-1]:42s} moy {np.mean(times):5.1f}s | "
          f"{n_dnf} DNF | {detail}", flush=True)
