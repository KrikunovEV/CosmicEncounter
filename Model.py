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


class MLP2(nn.Module):

    def __init__(self, state_space: int, action_space: int, agent_id: int, negotiation_map):
        super(MLP2, self).__init__()

        self.negotiation_map = negotiation_map
        self.dummy_tensor = torch.zeros((1, 10))
        self.agent_id = agent_id

        self.mlp = nn.Sequential(
            nn.Linear(110 + 50, 110 + 50),
            nn.ReLU(),
            nn.Linear(110 + 50, 64),
            nn.ReLU(),
        )

        self.mlp_neg = nn.Sequential(
            nn.Linear(110 + 50, 50),
            nn.Tanh(),
        )

        self.policy = nn.Linear(64, action_space)
        self.value = nn.Linear(64, 1)
        self.negotiation = nn.Sequential(
            nn.Linear(64, 10),
            nn.Tanh()
        )

    def forward(self, data):
        negotiation = data[1]
        negotiation = negotiation + torch.randn(negotiation.size()) * 0.1
        for i in range(5):
            if i not in self.negotiation_map or self.agent_id not in self.negotiation_map:
                negotiation = torch.cat([negotiation[:i], self.dummy_tensor, negotiation[i + 1:]], 0)
        negotiation = negotiation.view(-1)
        obs = torch.Tensor(data[0])

        negotiation_input = torch.cat((obs, negotiation), dim=-1)
        negotiation_output = self.mlp_neg(negotiation_input) * negotiation

        embedding = self.mlp(torch.cat((obs, negotiation_output), dim=-1))
        logits = self.policy(embedding)
        value = self.value(embedding)
        negotiation = self.negotiation(embedding.detach())
        return logits, value, negotiation
