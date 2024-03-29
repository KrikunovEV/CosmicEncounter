from Model import MLP2
import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim


class Agent:
    def __init__(self, agent_id: int, nrof_players: int, negotiation_map):
        self.agent_id = agent_id
        self.original_agent_id = agent_id
        self.model = MLP2(state_space=110 + (nrof_players - 1) * 10, action_space=14,
                          agent_id=agent_id, negotiation_map=negotiation_map)
        self.optim = optim.SGD(self.model.parameters(), lr=0.00001)
        self.logs, self.entropies, self.values, self.rewards = [], [], [], []
        self.losses, self.episode_mean_values = [], []
        self.grad_W, self.grad_b = [], []
        self.parties_won = 0
        self.reward_cum = [0]

    def __call__(self, obs, action_type, available_actions):
        logits, value, negotiation = self.model(obs)
        entropy = -(functional.log_softmax(logits, dim=-1) * functional.softmax(logits, dim=-1)).sum()

        logits = logits[available_actions]
        policy = functional.softmax(logits, dim=-1)

        #action_id = np.random.choice(policy.shape[0], 1, p=policy.detach().numpy())[0]
        action_id = policy.argmax()
        prob = policy[action_id]

        self.logs.append(torch.log(prob))
        self.entropies.append(entropy)
        self.values.append(value)

        return available_actions[action_id], negotiation

    def get_negotiation(self, obs):
        _, _, negotiation = self.model(obs)
        return negotiation

    def reward(self, reward):
        self.rewards.append(reward)

    def reward_win(self, reward):
        self.parties_won += 1
        self.rewards[-1] = reward
        self.reward_cum.append(self.reward_cum[-1] + reward)

    def reward_loose(self):
        self.reward_cum.append(self.reward_cum[-1])

    def train(self, obs, last: bool):
        _, G, _ = self.model(obs)
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
        loss.backward(retain_graph=not last)
        self.optim.step()

        self.grad_W.append(torch.norm(self.model.negotiation[0].weight.grad))
        self.grad_b.append(torch.norm(self.model.negotiation[0].bias.grad))

        self.episode_mean_values.append(torch.mean(torch.Tensor(self.values)))
        self.values, self.entropies, self.logs, self.rewards = [], [], [], []

    def save_agent_state(self, directory: str, episode_encounters: list):
        state = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'losses': self.losses,
            'mean_values': self.episode_mean_values,
            'parties_won': self.parties_won,
            'reward_cum': self.reward_cum,
            'episode_encounters': episode_encounters,
            'agent_id': self.original_agent_id,
            'grad_W': self.grad_W,
            'grad_b': self.grad_b
        }
        torch.save(state, directory + str(self.original_agent_id) + '.pt')

    def load_agent_state(self, path: str):
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        self.optim.load_state_dict(state['optim_state'])
        self.losses = state['losses']
        self.episode_mean_values = state['mean_values']
        self.parties_won = state['parties_won']
        self.reward_cum = state['reward_cum']
        self.original_agent_id = state['agent_id']
        return state['episode_encounters']
