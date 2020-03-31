import numpy as np
from dataclasses import dataclass


class Environment:

    @dataclass(frozen=True)
    class Const:
        MAX_SHIPS_PER_PLANET: int = 4
        MORPH_CARD: int = -1
        NEGOTIATE_CARD: int = 0
        CARDS_IN_HAND: int = 5

    @dataclass
    class Planet:
        player_home_id: int = -1
        ships: np.ndarray = np.array([], dtype=np.uint8)

        def __str__(self):
            return '    player_home_id: ' + str(self.player_home_id) + '\n    ships: ' + str(self.ships)

    def __init__(self, nrof_players: int, nrof_planets_per_player: int):
        self.nrof_players = nrof_players
        self.nrof_planets_per_player = nrof_planets_per_player

        self.planets = []
        for player_id in range(nrof_players):
            for planet_id in range(nrof_planets_per_player):
                ships = np.zeros(nrof_players)
                ships[player_id] = self.Const.MAX_SHIPS_PER_PLANET
                self.planets.append(self.Planet(player_id, ships))

        self.warp = self.Planet(ships=np.zeros(nrof_players))

        self.deck = []
        self.deck += [self.Const.MORPH_CARD] * 2
        self.deck += [self.Const.NEGOTIATE_CARD] * 15
        self.deck += [1] + [4] * 4 + [5] + [6] * 7 + [7] + [8] * 7 + [9] + [10] * 4 + [11] + [12] * 2 + [13] + [14] * 2\
                     + [15] + [20] * 2 + [23] + [30] + [40]
        np.random.shuffle(self.deck)

        self.player_hands = []
        for player_id in range(nrof_players):
            self.player_hands.append(self.deck[:self.Const.CARDS_IN_HAND])
            self.deck = self.deck[self.Const.CARDS_IN_HAND:]


    def __str__(self):
        desc = 'Cosmic Encounter environment:\n\n'
        for id, planet in enumerate(self.planets):
            desc += 'planet ' + str(id) + ':\n' + str(planet) + '\n'
        desc += '\nwarp:\n' + str(self.warp)
        desc += '\n\ndeck: ' + str(self.deck) + '\n\n'
        for id, hand in enumerate(self.player_hands):
            desc += 'player ' + str(id) + ' hand: ' + str(hand) + '\n'
        return desc


    