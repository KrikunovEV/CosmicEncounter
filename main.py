from CosmicEncounter import Environment

ships_on_planets = [(0, 5), (1, 7), (2, 1)]
print(min(ships_on_planets, key=lambda x:x[1]))

env = Environment(nrof_players=5, nrof_planets_per_player=3)
print(env.reset())
for i in range(5):
    print(env.whose_turn())
    env.action(0)
print(env)