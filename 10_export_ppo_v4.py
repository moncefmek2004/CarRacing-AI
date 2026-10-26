"""Usage : python 10_export_ppo_v4.py [zip] [sortie.pt]"""
import copy
import sys

import torch
from stable_baselines3 import PPO

from driver_ppo_v4 import DriverV4

src = sys.argv[1] if len(sys.argv) > 1 else "models/carracing_ppo_v4_final.zip"
dst = sys.argv[2] if len(sys.argv) > 2 else "models/carracing_ppo_v4_driver.pt"

p = PPO.load(src, device="cpu").policy.eval()
drv = DriverV4(
    cnn=copy.deepcopy(p.features_extractor.cnn),
    linear=copy.deepcopy(p.features_extractor.linear),
    policy_net=copy.deepcopy(p.mlp_extractor.policy_net),
    action_net=copy.deepcopy(p.action_net),
).eval()
torch.save(drv, dst)
print(f"Exporte : {dst}")
