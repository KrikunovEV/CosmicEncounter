from CosmicEncounter import Environment
from Agent import Agent
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt


players = 5
env = Environment(nrof_players=players, nrof_planets_per_player=3)
agents = [Agent(agent_id) for agent_id in range(players)]
agents[0].load_agent_state('models_5000/3.pt')
agents[1].load_agent_state('models_1000/0.pt')
agents[0].parties_won = 0
agents[1].parties_won = 0

values = [[], [], [], []]

for episode in range(1000):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()

    while not terminal:
        agent_id = env.whose_turn()[0]

        if agent_id != 4:
            action_id = agents[agent_id](obs, env.action_type(), env.available_actions())
            obs, terminal, winners, reward = env.action(action_id)
        else:
            obs, terminal, winners, reward = env.action(env.available_actions()[np.random.randint(
                len(env.available_actions()))])
        agents[agent_id].reward(reward)
        if terminal:
            for agent_id in winners:
                agents[agent_id].reward_win(reward)

    for agent_id in range(players - 1):
        values[agent_id].append(Tensor(agents[agent_id].values).mean().item())
        agents[agent_id].values = []

for agent_id in range(players):
    print('Agent id: ' + str(agent_id) + '; wins: ' + str(agents[agent_id].parties_won))

fig, ax = plt.subplots(1, 4, figsize=(16, 9))
for agent_id in range(players - 1):
    if agent_id == 0:
        ax[agent_id].set_title('5000 episodes: ' + str(agents[agent_id].parties_won))
    elif agent_id == 1:
        ax[agent_id].set_title('1000 episodes: ' + str(agents[agent_id].parties_won))
    else:
        ax[agent_id].set_title('Randomly initialized: ' + str(agents[agent_id].parties_won))
    ax[agent_id].plot(values[agent_id])
    ax[agent_id].set_xlabel('episode')
    ax[agent_id].set_ylabel('mean value')
plt.show()
