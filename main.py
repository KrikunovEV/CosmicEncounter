from CosmicEncounter import Environment
from Agent import Agent
import numpy as np


players = 5
negotiation_map = [0, 1, 2, 3, 4]

env = Environment(nrof_players=players, nrof_planets_per_player=3)
agents = [Agent(agent_id, players, negotiation_map) for agent_id in range(players)]
episode_encounters = []

for episode in range(2350):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()
    episode_encounters.append(0)

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

    for agent_id in range(players):
        agents[agent_id].train(obs, True if agent_id == players - 1 else False)

for agent_id in range(players):
    agents[agent_id].save_agent_state(directory='everyone_grad_2350/', episode_encounters=episode_encounters)
