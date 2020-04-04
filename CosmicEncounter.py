import numpy as np
from dataclasses import dataclass
from typing import List


class Environment:

    @dataclass(init=False, frozen=True)
    class Const:
        MAX_SHIPS_PER_PLANET: int = 4
        MORPH_CARD: int = -1
        NEGOTIATE_CARD: int = 0
        CARDS_IN_HAND: int = 5
        WARP_HOME_ID: int = -1
        GATE_HOME_ID: int = -2

    class Planet:
        def __init__(self, player_home_id: int, ships: List[int]):
            self.player_home_id = player_home_id
            self.ships = ships

        def __str__(self):
            return '    player_home_id: ' + str(self.player_home_id) + '\n    ships: ' + str(self.ships)

    class Warp(Planet):
        def __init__(self, player_home_id: int, ships: List[int]):
            super().__init__(player_home_id, ships)
            self.planet_counter = [0 for _ in range(len(ships))]

        def __str__(self):
            return '    planet_counter: ' + str(self.planet_counter) + '\n' + super().__str__()

    def __init__(self, nrof_players: int, nrof_planets_per_player: int):
        self.nrof_players = nrof_players
        self.nrof_planets_per_player = nrof_planets_per_player

    def reset(self):
        self.planets = []
        for player_id in range(self.nrof_players):
            for planet_id in range(self.nrof_planets_per_player):
                ships = [0 for _ in range(self.nrof_players)]
                ships[player_id] = self.Const.MAX_SHIPS_PER_PLANET
                self.planets.append(self.Planet(player_id, ships))

        self.warp = self.Warp(self.Const.WARP_HOME_ID, [0 for _ in range(self.nrof_players)])

        self.gate = self.Planet(self.Const.GATE_HOME_ID, [0 for _ in range(self.nrof_players)])

        self.deck = []
        self.deck += [self.Const.MORPH_CARD] * 2
        self.deck += [self.Const.NEGOTIATE_CARD] * 15
        self.deck += [1] + [4] * 4 + [5] + [6] * 7 + [7] + [8] * 7 + [9] + [10] * 4 + [11] + [12] * 2 + [13] + [14] * 2\
                     + [15] + [20] * 2 + [23] + [30] + [40]
        np.random.shuffle(self.deck)

        self.player_hands = []
        for player_id in range(self.nrof_players):
            self.player_hands.append(self.deck[:self.Const.CARDS_IN_HAND])
            self.deck = self.deck[self.Const.CARDS_IN_HAND:]

        self.offender_id = np.random.randint(0, self.nrof_players)
        self.__reset_round()

        return self.__make_observation()

    def action(self, nrof_ships: int):
        player_id = self.player_turns.pop(0)

        if len(self.player_turns) == 0:
            # __do_encounter():
            # ...

            # move to __do_encounter ?
            self.offender_id = (self.offender_id + 1) if (self.offender_id + 1) < self.nrof_players else 0

            self.__reset_round()
            pass
        return self.__make_observation()

    def whose_turn(self):
        return self.player_turns

    def __make_observation(self) -> np.ndarray:
        observation = [self.offender_id, self.defender_id]
        for planet in self.planets:
            observation += [planet.player_home_id] + planet.ships
        observation += [self.warp.player_home_id] + self.warp.ships
        observation += [self.gate.player_home_id] + self.gate.ships
        observation += self.player_hands[self.offender_id]
        return np.array(observation, dtype=np.float_)

    def __reset_round(self):
        self.defender_id = np.random.choice([i for i in range(self.nrof_players) if i != self.offender_id])

        planets_to_offend = [i for i in range(len(self.planets)) if self.planets[i].player_home_id != self.offender_id]
        self.defender_planet_id = np.random.choice(planets_to_offend)

        self.player_turns = [i for i in range(self.nrof_players)
            if i != self.offender_id and i != self.defender_id and self.planets[self.defender_planet_id].ships[i] == 0]
        next_id = self.offender_id + 1
        if next_id == self.defender_id:
            next_id = self.defender_id + 1
        if next_id == self.nrof_players:
            next_id = 0
        next_id = self.player_turns.index(next_id)
        self.player_turns = self.player_turns[next_id:] + self.player_turns[:next_id]
        self.player_turns = [self.offender_id, self.defender_id] + self.player_turns

        if self.warp.ships[self.offender_id] != 0:
            self.warp.ships[self.offender_id] -= 1
            ships_on_planets = self.__get_ships_on_planets_by_player_id(self.offender_id)
            (planet_id, nrof_ships) = min(ships_on_planets, key=lambda x:x[1])
            self.planets[planet_id].ships[self.offender_id] += 1

    def __get_ships_on_planets_by_player_id(self, player_id):
        player_planets = []
        for id, planet in enumerate(self.planets):
            if planet.ships[player_id] != 0 or planet.player_home_id == player_id:
                player_planets.append((id, planet.ships[player_id]))
        return player_planets

    def __str__(self):
        desc = 'Cosmic Encounter environment:\n\n'
        for id, planet in enumerate(self.planets):
            desc += 'planet ' + str(id) + ':\n' + str(planet) + '\n'
        desc += '\nwarp:\n' + str(self.warp)
        desc += '\n\ngate:\n' + str(self.gate)
        desc += '\n\ndeck: ' + str(self.deck) + '\n\n'
        for id, hand in enumerate(self.player_hands):
            desc += 'player ' + str(id) + ' hand: ' + str(hand) + '\n'
        return desc
