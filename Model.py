import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, state_space: int, action_space: int):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_space, state_space),
            nn.ReLU(),
            nn.Linear(state_space, 64),
            nn.ReLU(),
        )

        self.policy = nn.Linear(64, action_space)
        self.value = nn.Linear(64, 1)

    def forward(self, data):
        mlp_data = self.mlp(torch.Tensor(data))
        logits = self.policy(mlp_data)
        value = self.value(mlp_data)
        return logits, value

    def __preprocess(self, data):
        return torch.log(data)
