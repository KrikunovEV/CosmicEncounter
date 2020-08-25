import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, state_space: int, action_space: int, agent_id: int, negotiation_map):
        super(MLP, self).__init__()

        self.negotiation_map = negotiation_map
        self.agent_id = agent_id
        self.dummy_tensor = torch.zeros(10, requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(110, 110),
            nn.ReLU(),
            nn.Linear(110, 64),
            nn.ReLU(),
        )

        self.mlp_neg = nn.Sequential(
            nn.Linear(40, 40),
            nn.Tanh(),
        )

        self.policy = nn.Linear(104, action_space)
        self.value = nn.Linear(104, 1)
        self.negotiation = nn.Sequential(
            nn.Linear(104, 10),
            nn.Tanh()
        )

    def forward(self, data):
        negotiation = torch.empty(0)
        for i in range(5):
            if i == self.agent_id:
                continue
            elif i in self.negotiation_map[self.agent_id]:
                negotiation = torch.cat((negotiation, data[1][i]))
            else:
                negotiation = torch.cat((negotiation, self.dummy_tensor))
        embeddings = torch.cat((self.mlp(torch.Tensor(data[0])), self.mlp_neg(negotiation)), dim=-1)

        logits = self.policy(embeddings)
        value = self.value(embeddings)
        negotiation = self.negotiation(embeddings.detach())
        return logits, value, negotiation

    def __preprocess(self, data):
        return torch.log(data)
