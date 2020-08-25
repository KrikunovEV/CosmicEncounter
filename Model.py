import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, state_space: int, action_space: int, agent_id: int, negotiation_map):
        super(MLP, self).__init__()

        self.negotiation_map = negotiation_map
        self.agent_id = agent_id
        self.dummy_tensor = torch.zeros(10, requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(state_space, state_space),
            nn.ReLU(),
            nn.Linear(state_space, 64),
            nn.ReLU(),
        )

        self.policy = nn.Linear(64, action_space)
        self.value = nn.Linear(64, 1)
        self.negotiation = nn.Sequential(
            nn.Linear(64, 10),
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
        state = torch.cat((torch.Tensor(data[0]), negotiation), dim=-1)

        embeddings = self.mlp(torch.Tensor(state))
        logits = self.policy(embeddings)
        value = self.value(embeddings)
        negotiation = self.negotiation(embeddings.detach())
        return logits, value, negotiation

    def __preprocess(self, data):
        return torch.log(data)
