from CosmicEncounter import Environment
from Agent import Agent


players = 5
env = Environment(nrof_players=players, nrof_planets_per_player=3)
agents = [Agent(agent_id) for agent_id in range(players)]

for episode in range(1000):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()

    while not terminal:
        agent_id = env.whose_turn()[0]
        action_id = agents[agent_id](obs, env.action_type(), env.available_actions())
        obs, terminal, winners, reward = env.action(action_id)
        agents[agent_id].reward(reward)

        if terminal:
            for agent_id in winners:
                agents[agent_id].reward_win(reward)

    for agent_id in range(players):
        agents[agent_id].train(obs)

for agent_id in range(players):
    agents[agent_id].save_agent_state(directory='models_1000/')
