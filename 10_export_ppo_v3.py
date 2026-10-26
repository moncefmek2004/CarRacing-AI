"""Exporte un modèle PPO (.zip SB3) vers un .pt autonome.

Usage : python 10_export_ppo_v3.py [zip] [sortie.pt]
"""
import copy
import sys

import torch
from stable_baselines3 import PPO

from driver_ppo import Driver

src = sys.argv[1] if len(sys.argv) > 1 else "models/carracing_ppo_v3_final.zip"
dst = sys.argv[2] if len(sys.argv) > 2 else "models/carracing_ppo_v3_driver.pt"

model = PPO.load(src, device="cpu")
p = model.policy.eval()

driver = Driver(
    cnn=copy.deepcopy(p.features_extractor.cnn),
    linear=copy.deepcopy(p.features_extractor.linear),
    policy_net=copy.deepcopy(p.mlp_extractor.policy_net),
    action_net=copy.deepcopy(p.action_net),
    skip=2,
    n_stack=4,
).eval()

torch.save(driver, dst)
print(f"Exporte : {dst}")
