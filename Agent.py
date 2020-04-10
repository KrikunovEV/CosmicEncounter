from Model import MLP
import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.model = MLP(state_space=110, action_space=14)
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.logs, self.entropies, self.values, self.rewards = [], [], [], []
        self.losses = []
        self.parties_won = 0

    def __call__(self, obs, action_type, available_actions):
        logits, value = self.model(obs)
        entropy = -(functional.log_softmax(logits, dim=-1) * functional.softmax(logits, dim=-1)).sum()

        logits = logits[available_actions]
        policy = functional.softmax(logits, dim=-1)

        action_id = np.random.choice(policy.shape[0], 1, p=policy.detach().numpy())[0]
        prob = policy[action_id]

        self.logs.append(torch.log(prob))
        self.entropies.append(entropy)
        self.values.append(value)

        return available_actions[action_id]

    def reward(self, reward):
        self.rewards.append(reward)

    def reward_win(self, reward):
        self.parties_won += 1
        self.rewards[-1] = reward

    def train(self, obs):
        _, G = self.model(obs)
        G = G.detach().item()

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + 0.99 * G
            advantage = G - self.values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (advantage.detach() * self.logs[i] + 0.001 * self.entropies[i])

        loss = policy_loss + 0.5 * value_loss
        self.losses.append(loss.item())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.values, self.entropies, self.spatial_entropies, self.logs, self.rewards = [], [], [], [], []

    def save_agent_state(self):
        state = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'losses': self.losses,
            'parties_won': self.parties_won
        }
        torch.save(state, 'models/' + str(self.agent_id) + '.pt')

    def load_agent_state(self, path: str):
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        self.optim.load_state_dict(state['optim_state'])
        self.losses = state['losses']
        self.parties_won = state['parties_won']
