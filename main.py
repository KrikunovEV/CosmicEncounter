from CosmicEncounter import Environment
from Agent import Agent


players = 5
#negotiation_map = [[], [], [], [], []]
#negotiation_map = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
#negotiation_map = [[1], [0], [], [], []]
negotiation_map = [[1, 2], [0, 2], [0, 1], [], []]

env = Environment(nrof_players=players, nrof_planets_per_player=3)
agents = [Agent(agent_id, players, negotiation_map) for agent_id in range(players)]

for episode in range(2500):
    print('Episode:', episode)
    obs, terminal, winners, reward = env.reset()

    while not terminal:
        agent_id = env.whose_turn()[0]
        action_id, negotiation = agents[agent_id](obs, env.action_type(), env.available_actions())
        obs, terminal, winners, reward = env.action(action_id, negotiation)
        agents[agent_id].reward(reward)

        if terminal:
            for agent_id in winners:
                agents[agent_id].reward_win(reward)

    for agent_id in range(players):
        agents[agent_id].train(obs, True if agent_id == players - 1 else False)

for agent_id in range(players):
    agents[agent_id].save_agent_state(directory='three_new/')
