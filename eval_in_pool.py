from CosmicEncounter import Environment
from Agent import Agent
import numpy as np
from torch import zeros
import matplotlib.pyplot as plt


players = 5
env = Environment(nrof_players=players, nrof_planets_per_player=3)
negotiation = [0, 1, 2, 3]
agents_pool_red = []
sampling_round = 25
for agent_id in range(20):
    agents_pool_red.append(Agent(agent_id, players, negotiation))
    agents_pool_red[-1].load_agent_state('pool_red/' + str(agent_id) + '.pt')
    agents_pool_red[-1].parties_won = 0
    agents_pool_red[-1].reward_cum = [0]

agents_pool_blue = []
for agent_id in range(20):
    agents_pool_blue.append(Agent(agent_id, players, negotiation))
    agents_pool_blue[-1].load_agent_state('pool_blue/' + str(agent_id) + '.pt')
    agents_pool_blue[-1].parties_won = 0
    agents_pool_blue[-1].reward_cum = [0]

newbie_agent = Agent(20, players, negotiation)
newbie_agent.load_agent_state('pool_red/' + '20.pt')
newbie_agent.parties_won = 0
newbie_agent.reward_cum = [0]

newbie_agent2 = Agent(21, players, negotiation)
newbie_agent2.load_agent_state('nobody_2350/' + '2.pt')
newbie_agent2.parties_won = 0
newbie_agent2.reward_cum = [0]

parties_won = [0, 0, 0, 0, 0]
scores = [[0], [0], [0], [0], [0]]

for episode in range(1000):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()

    if episode % sampling_round == 0:
        if episode != 0:
            for id, agent in enumerate(agents):  # [:players-2]
                parties_won[id] += agent.parties_won
                agent.parties_won = 0
                scores[id] += agent.reward_cum[1:]
        agents = np.random.choice(agents_pool_red, 4, replace=False)
        agent_blue = np.random.choice(agents_pool_blue, 1, replace=False)
        agents = np.hstack((agents, agent_blue))
        for id, agent in enumerate(agents):  # [:players-2]
            agent.agent_id = id
            agent.reward_cum = [scores[id][-1]]

    while not terminal:
        agent_id = env.whose_turn()[0]
        action_id, negotiation = agents[agent_id](obs, env.action_type(), env.available_actions())
        obs, terminal, winners, reward = env.action(action_id, negotiation)
        agents[agent_id].reward(reward)

        new_negotiations = []
        for i in range(players):
            if i == agent_id:
                continue
            negotiation = agents[i].get_negotiation(obs)
            new_negotiations.append([i, negotiation])
        env.update_negotiation(negotiations=new_negotiations)

        if terminal:
            for agent_id in range(players):
                if agent_id in winners:
                    agents[agent_id].reward_win(reward)
                else:
                    agents[agent_id].reward_loose()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for agent_id in range(players):
    j = agent_id % 3
    i = agent_id // 3
    '''
    if agent_id == 3:
        ax[i][j].set_title('Newbie agent. wins:' + str(agents[agent_id].parties_won) + '; score = '
                                                                                                      '%.2f' % agents[
            agent_id].reward_cum[-1])
        ax[i][j].plot(agents[agent_id].reward_cum)
        print('Agent id: ' + str(agent_id) + '; wins: ' + str(agents[agent_id].parties_won) + '; score = %.2f' %
              agents[agent_id].reward_cum[-1])
    elif agent_id == 4:
        ax[i][j].set_title('Silent agent. wins:' + str(agents[agent_id].parties_won) + '; score = '
                                                                                       '%.2f' % agents[
                               agent_id].reward_cum[-1])
        ax[i][j].plot(agents[agent_id].reward_cum)
        print('Agent id: ' + str(agent_id) + '; wins: ' + str(agents[agent_id].par
        ties_won) + '; score = %.2f' %
              agents[agent_id].reward_cum[-1])
    else:
    '''
    ax[i][j].set_title('Pool agent. wins:' + str(parties_won[agent_id]) + '; score = %.2f' % scores[agent_id][-1])
    ax[i][j].plot(scores[agent_id])
    print('Agent id: ' + str(agent_id) + '; wins: ' + str(parties_won[agent_id]) + '; score = %.2f' % scores[agent_id][-1])

    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('cumulative reward')
    ax[i][j].set_yticks(np.arange(0, 501, 50))
fig.tight_layout()
plt.savefig('eval/4_red_pool_VS_1_blue_pool.png')
