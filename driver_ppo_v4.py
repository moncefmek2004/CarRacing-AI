"""Driver v4 : sortie bornée par tanh, 2 actions (direction, gaz),
gaz plafonné à MAX_GAS, frein à 0. Prétraitement identique à l'entraînement."""
import numpy as np
import torch
import torch.nn as nn

MAX_GAS = 0.5


class DriverV4(nn.Module):
    def __init__(self, cnn, linear, policy_net, action_net,
                 skip=2, n_stack=4, max_gas=MAX_GAS):
        super().__init__()
        self.cnn = cnn
        self.linear = linear
        self.policy_net = policy_net
        self.action_net = action_net
        self.skip = skip
        self.n_stack = n_stack
        self.max_gas = max_gas
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114]))
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
            z = self.policy_net(self.linear(self.cnn(self.stack[None])))
            a = torch.tanh(self.action_net(z)[0])
            gas = (a[1] + 1.0) / 2.0 * self.max_gas
            self.last_action = torch.stack([a[0], gas, torch.zeros(())])
        self.t += 1
        return self.last_action.numpy().astype(np.float32)
