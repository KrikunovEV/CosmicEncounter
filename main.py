from CosmicEncounter import Environment
import numpy as np

env = Environment(nrof_players=5, nrof_planets_per_player=3)


obs, terminal, players, reward = env.reset()
while not terminal:
    player_id = env.whose_turn()[0]
    (ships, cards) = env.available_actions()
    action_type = env.action_type()
    action_id = np.random.choice(ships if action_type == Environment.Const.ACTION_SHIP else cards)
    '''
    print('Offender:', env.who_offender())
    print('Defender:', env.who_defender())
    print('Player turns:', player_id)
    print('Available actions:', ships, cards)
    print('action is', action_id)
    '''
    obs, terminal, players, reward = env.action(action_id)

