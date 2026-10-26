"""Driver PPO : modèle autonome (PyTorch pur) qui prend l'image brute
96x96x3 et renvoie [steering, gas, brake]. Le prétraitement (gris,
frame skip, frame stack) est fait ICI, comme à l'entraînement.
Modèle avec état : le recharger (ou reset()) à chaque épisode.
"""
import numpy as np
import torch
import torch.nn as nn


class Driver(nn.Module):
    def __init__(self, cnn, linear, policy_net, action_net, skip=2, n_stack=4):
        super().__init__()
        self.cnn = cnn
        self.linear = linear
        self.policy_net = policy_net
        self.action_net = action_net
        self.skip = skip
        self.n_stack = n_stack
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114]))
        self.register_buffer("low", torch.tensor([-1.0, 0.0, 0.0]))
        self.register_buffer("high", torch.tensor([1.0, 1.0, 1.0]))
        self.reset()

    def reset(self):
        self.t = 0
        self.stack = torch.zeros(self.n_stack, 96, 96)
        self.last_action = torch.tensor([0.0, 0.0, 0.0])

    @torch.no_grad()
    def forward(self, obs):
        x = torch.as_tensor(np.asarray(obs))
        if x.ndim == 4:
            x = x[0]
        if self.t % self.skip == 0:
            gray = (x.float() @ self.gray_w).to(torch.uint8).float() / 255.0
            self.stack = torch.cat([self.stack[1:], gray[None]], dim=0)
            z = self.cnn(self.stack[None])
            z = self.linear(z)
            z = self.policy_net(z)
            a = self.action_net(z)[0]
            self.last_action = torch.max(torch.min(a, self.high), self.low)
        self.t += 1
        return self.last_action.numpy().astype(np.float32)
