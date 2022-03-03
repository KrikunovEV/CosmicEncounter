from CosmicEncounter import Environment
from Agent import Agent
import numpy as np


players = 5
negotiation_map = [0, 1, 2, 3, 4]

env = Environment(nrof_players=players, nrof_planets_per_player=3)
agents_pool = []
episode_encounters = []
sampling_round = 25
path = 'pool_red/'
for agent_id in range(20):
    agents_pool.append(Agent(agent_id, players, negotiation_map))
    agents_pool[-1].load_agent_state(path + str(agent_id) + '.pt')
    agents_pool[-1].parties_won = 0
    agents_pool[-1].reward_cum = [0]

newbie_agent = Agent(20, players, negotiation_map)

for episode in range(2350):  # 10 000
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()
    episode_encounters.append(0)

    if episode % sampling_round == 0:
        agents = np.random.choice(agents_pool, 4, replace=False)
        for agent in agents:
            agent.logs, agent.entropies, agent.values, agent.rewards = [], [], [], []
            agent.losses, agent.episode_mean_values = [], []
            agent.parties_won = 0
            agent.reward_cum = [0]
        agents = np.hstack((agents, newbie_agent))
        for id, agent in enumerate(agents):
            agent.agent_id = id

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

        if len(env.player_turns) == 1:
            episode_encounters[-1] += 1

        if terminal:
            for agent_id in range(players):
                if agent_id in winners:
                    agents[agent_id].reward_win(reward)
                else:
                    agents[agent_id].reward_loose()

    #for agent_id in range(players):
    #    agents[agent_id].train(obs, True if agent_id == players - 1 else False)
    newbie_agent.train(obs, False)

#for agent_id in range(20):
#    agents_pool[agent_id].save_agent_state(directory=path, episode_encounters=episode_encounters)

newbie_agent.save_agent_state(directory=path, episode_encounters=episode_encounters)
