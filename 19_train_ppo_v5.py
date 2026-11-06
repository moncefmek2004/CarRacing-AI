"""PPO v5 : reprise depuis BEST_v4_300k, melange piste(s) focus + pistes aleatoires,
arret si aucune tuile gagnee, evaluation deterministe avec sauvegarde du meilleur.

Usage : python 19_train_ppo_v5.py [etapes] [seed_focus1 seed_focus2 ...]
Ex.   : python 19_train_ppo_v5.py 500000 43
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CallbackList, CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecMonitor)

from driver_ppo_v4 import MAX_GAS

N_ENVS = 8
SKIP = 2
N_STACK = 4
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
FOCUS_SEEDS = [int(s) for s in sys.argv[2:]] or [43]
P_FOCUS = 0.5
EVAL_SEEDS = [1, 2, 3, 4, 5, 6]
START_MODEL = "models/BEST_v4_300k.zip"
GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_gray(obs):
    g = (obs.astype(np.float32) @ GRAY_W).astype(np.uint8)
    return g[..., None]


def on_grass(obs):
    p = obs[66:78, 44:52].astype(np.int16)
    green = (p[..., 1] - p[..., 0] > 40) & (p[..., 1] - p[..., 2] > 40)
    return green.mean() > 0.5


class CarWrapper(gym.Wrapper):
    def __init__(self, env, skip=SKIP, grass_penalty=0.02, grass_limit=100,
                 idle_limit=100, focus_seeds=None, p_focus=0.0, seed_cycle=None):
        super().__init__(env)
        self.skip = skip
        self.grass_penalty = grass_penalty
        self.grass_limit = grass_limit
        self.idle_limit = idle_limit
        self.focus_seeds = focus_seeds or []
        self.p_focus = p_focus
        self.seed_cycle = seed_cycle
        self.rng = np.random.default_rng()
        self.cycle_i = 0
        self.grass_steps = 0
        self.idle_steps = 0
        self.observation_space = spaces.Box(0, 255, (96, 96, 1), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:
            if self.seed_cycle:
                kwargs["seed"] = self.seed_cycle[self.cycle_i % len(self.seed_cycle)]
                self.cycle_i += 1
            elif self.focus_seeds and self.rng.random() < self.p_focus:
                kwargs["seed"] = int(self.rng.choice(self.focus_seeds))
            else:
                kwargs["seed"] = int(self.rng.integers(1000, 2**31 - 1))
        obs, info = self.env.reset(**kwargs)
        self.grass_steps = 0
        self.idle_steps = 0
        return to_gray(obs), info

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        env_action = np.array([a[0], (a[1] + 1.0) / 2.0 * MAX_GAS, 0.0],
                              dtype=np.float32)
        raw = 0.0
        for _ in range(self.skip):
            obs, r, terminated, truncated, info = self.env.step(env_action)
            raw += r
            if terminated or truncated:
                break
        total = raw
        # une tuile gagnee donne environ +3.5 ; sinon -0.1 par pas de jeu
        self.idle_steps = 0 if raw > 0 else self.idle_steps + 1
        if on_grass(obs):
            self.grass_steps += 1
            total -= self.grass_penalty
        else:
            self.grass_steps = 0
        if self.grass_steps >= self.grass_limit:
            terminated = True
        if self.idle_steps >= self.idle_limit:
            terminated = True
            total -= 10.0
        return to_gray(obs), total, terminated, truncated, info


def linear_schedule(initial):
    return lambda progress_remaining: progress_remaining * initial


if __name__ == "__main__":
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    venv = make_vec_env(
        "CarRacing-v3",
        n_envs=N_ENVS,
        wrapper_class=CarWrapper,
        wrapper_kwargs=dict(focus_seeds=FOCUS_SEEDS, p_focus=P_FOCUS),
        env_kwargs={"max_episode_steps": 3000},
        vec_env_cls=SubprocVecEnv,
    )
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="last")

    eval_env = make_vec_env(
        "CarRacing-v3",
        n_envs=1,
        wrapper_class=CarWrapper,
        wrapper_kwargs=dict(seed_cycle=EVAL_SEEDS),
        env_kwargs={"max_episode_steps": 3000},
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")

    model = PPO.load(START_MODEL, env=venv, device="auto",
                     tensorboard_log="./logs/")
    model.learning_rate = linear_schedule(3e-5)
    model._setup_lr_schedule()

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1, 25_000 // N_ENVS),
                           save_path="./models/checkpoints/",
                           name_prefix="carracing_ppo_v5"),
    ])

    print(f"PPO v5 : {TOTAL_STEPS} etapes | focus {FOCUS_SEEDS} | depart {START_MODEL}")
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks,
                    tb_log_name="PPO_CarRacing_v5")
    finally:
        model.save("models/carracing_ppo_v5_final")
        venv.close()
        eval_env.close()
        print("Modele : models/carracing_ppo_v5_final.zip")