from CosmicEncounter import Environment
from Agent import Agent
import numpy as np
from torch import zeros
import matplotlib.pyplot as plt


players = 5
env = Environment(nrof_players=players, nrof_planets_per_player=3)
negotiation_map_nobody = [[], [], [], [], []]
negotiation_map_everyone = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
negotiation_map_two = [[1], [0], [], [], []]
negotiation_map_three = [[1, 2], [0, 2], [0, 1], [], []]
negotiation_map_case1 = [[], [], [], [], []]
negotiation_map_case2 = [[1, 2], [0, 2], [0, 1], [], []]
agents = [Agent(0, players, negotiation_map_case1),
          Agent(1, players, negotiation_map_case1),
          Agent(2, players, negotiation_map_case1),
          Agent(3, players, negotiation_map_case1),
          Agent(4, players, negotiation_map_case1)]
agents[0].load_agent_state('everyone/3.pt')
agents[1].load_agent_state('nobody/4.pt')
agents[2].load_agent_state('nobody/4.pt')
agents[3].load_agent_state('nobody/4.pt')
agents[0].parties_won = 0
agents[1].parties_won = 0
agents[2].parties_won = 0
agents[3].parties_won = 0
agents[4].parties_won = 0

for episode in range(1000):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()

    while not terminal:
        agent_id = env.whose_turn()[0]

        if agent_id != 4:
            action_id, negotiation = agents[agent_id](obs, env.action_type(), env.available_actions())
            obs, terminal, winners, reward = env.action(action_id, negotiation)
        else:
            obs, terminal, winners, reward = env.action(env.available_actions()[np.random.randint(
                len(env.available_actions()))], zeros(10, requires_grad=False))
        agents[agent_id].reward(reward)
        if terminal:
            for agent_id in winners:
                agents[agent_id].reward_win(reward)

fig, ax = plt.subplots(1, 5, figsize=(16, 9))
for agent_id in range(players):
    print('Agent id: ' + str(agent_id) + '; wins: ' + str(agents[agent_id].parties_won) + '; score = ' + str(
        agents[agent_id].reward_cum[-1]))
    if agent_id == 0:
        ax[agent_id].set_title('Communication agent. wins:' + str(agents[agent_id].parties_won) + '; score = ' + str(
        agents[agent_id].reward_cum[-1]))
    elif agent_id != 4:
        ax[agent_id].set_title('Noncommunication agent. wins:' + str(agents[agent_id].parties_won) + '; score = ' + str(
        agents[agent_id].reward_cum[-1]))
    else:
        ax[agent_id].set_title('Random agent. wins:' + str(agents[agent_id].parties_won) + '; score = ' + str(
        agents[agent_id].reward_cum[-1]))
    ax[agent_id].plot(agents[agent_id].reward_cum)
    ax[agent_id].set_xlabel('episode')
    ax[agent_id].set_ylabel('cumulative reward')
    ax[agent_id].set_yticks(np.arange(0, 801, 100))
fig.tight_layout()
plt.savefig('eval/exp_1.png')
