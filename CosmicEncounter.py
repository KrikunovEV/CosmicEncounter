import numpy as np
from dataclasses import dataclass
from typing import List
import torch


class Environment:

    @dataclass(init=False, frozen=True)
    class Const:
        MAX_SHIPS_PER_PLANET: int = 4
        MORPH_CARD: int = -1
        NEGOTIATE_CARD: int = 0
        CARDS_IN_HAND: int = 5
        WARP_HOME_ID: int = -1
        GATE_HOME_ID: int = -2
        ACTION_SHIP: int = 0
        ACTION_CARD: int = 1
        PLANETS_TO_WIN: int = 5

    class Planet:
        def __init__(self, owner_id: int, ships: List[int]):
            self.owner_id = owner_id
            self.ships = ships

        def __str__(self):
            return '    player_home_id: ' + str(self.owner_id) + '\n    ships: ' + str(self.ships)

    class Warp(Planet):
        def __init__(self, nrof_players: int):
            super().__init__(Environment.Const.WARP_HOME_ID, [0] * nrof_players)
            self.planet_counter = [0] * nrof_players

        def __str__(self):
            return '    planet_counter: ' + str(self.planet_counter) + '\n' + super().__str__()

    class Gate(Planet):
        def __init__(self, nrof_players: int):
            super().__init__(Environment.Const.GATE_HOME_ID, [0] * nrof_players)
            self.offend_card = None
            self.defend_card = None
            self.player_side = [None] * nrof_players

        def __str__(self):
            return '    offend_card: ' + str(self.offend_card) + '\n    defend_card: ' + str(self.defend_card) +\
                   '\n    player_side: ' + str(self.player_side) + '\n' + super().__str__()

    def __init__(self, nrof_players: int, nrof_planets_per_player: int):
        self.nrof_players = nrof_players
        self.nrof_planets_per_player = nrof_planets_per_player

    # Resets the game state:
    # 1. Planets
    # 2. Warp
    # 3. Gate
    # 4. Deck
    # 5. Players' hand
    # 6. Random offender
    # 8. Negotiation observation
    # 7. Resets round
    def reset(self):
        self.planets = []
        for player_id in range(self.nrof_players):
            for planet_id in range(self.nrof_planets_per_player):
                ships = [0] * self.nrof_players
                ships[player_id] = self.Const.MAX_SHIPS_PER_PLANET
                self.planets.append(self.Planet(player_id, ships))

        self.warp = self.Warp(self.nrof_players)

        self.gate = self.Gate(self.nrof_players)

        self.drop = []
        self.deck = [self.Const.MORPH_CARD]
        self.deck += [self.Const.NEGOTIATE_CARD] * 15
        self.deck += [1] + [4] * 4 + [5] + [6] * 7 + [7] + [8] * 7 + [9] + [10] * 4 + [11] + [12] * 2 + [13] + [14] * 2\
                     + [15] + [20] * 2 + [23] + [30] + [40]
        np.random.shuffle(self.deck)

        self.player_hand = []
        for player_id in range(self.nrof_players):
            self.player_hand.append(self.deck[:self.Const.CARDS_IN_HAND])
            self.deck = self.deck[self.Const.CARDS_IN_HAND:]

        self.offender_id = np.random.randint(0, self.nrof_players)

        self.__reset_round()

        return self.__make_observation(), False, [], 0

    def action(self, action_id: int, negotiation):
        terminal, players, reward = False, [], 0
        action_type = self.action_type()
        player_id = self.player_turns.pop(0)

        self.negotiation_obs = torch.cat([self.negotiation_obs[:player_id], negotiation.view(1, 10),
                                          self.negotiation_obs[player_id + 1:]], 0)

        if action_type == self.Const.ACTION_SHIP:
            if action_id == 0 and player_id == self.offender_id:
                self.gate.player_side[player_id] = self.offender_id
            elif action_id != 0:
                side = self.offender_id if (action_id - 1) // self.Const.MAX_SHIPS_PER_PLANET == 0 else self.defender_id
                self.gate.player_side[player_id] = side

                ships = ((action_id - 1) % self.Const.MAX_SHIPS_PER_PLANET) + 1
                for i in range(ships):
                    ships_on_planets = self.__get_ships_on_planets_by_player_id(player_id)
                    (planet_id, nrof_ships) = max(ships_on_planets, key=lambda x: x[1])
                    self.planets[planet_id].ships[player_id] -= 1
                    self.gate.ships[player_id] += 1
        else:
            card = self.player_hand[player_id].pop(action_id - 9)
            self.drop.append(card)
            if player_id == self.offender_id:
                self.gate.offend_card = card
            else:
                self.gate.defend_card = card

        if len(self.player_turns) == 0:
            self.__do_encounter()
            self.__reset_round()
            terminal, players, reward = self.__check_win_condition()

        return self.__make_observation(), terminal, players, reward

    def update_negotiation(self, negotiations):
        for negotiation in negotiations:
            id = negotiation[0]
            self.negotiation_obs = torch.cat([self.negotiation_obs[:id], negotiation[1].view(1, 10),
                                              self.negotiation_obs[id + 1:]], 0)

    def whose_turn(self):
        return self.player_turns

    def who_offender(self):
        return self.offender_id

    def who_defender(self):
        return self.defender_id

    def action_type(self):
        return self.Const.ACTION_SHIP if len(self.player_turns) > 2 else self.Const.ACTION_CARD

    def available_actions(self):
        if self.action_type() == self.Const.ACTION_SHIP:
            ships = 0
            ships_on_planets = self.__get_ships_on_planets_by_player_id(self.player_turns[0])
            for (planet_id, nrof_ship) in ships_on_planets:
                ships += nrof_ship
            ships = self.Const.MAX_SHIPS_PER_PLANET if ships > self.Const.MAX_SHIPS_PER_PLANET else ships
            action_ind = np.arange(ships + 1)
            if self.player_turns[0] != self.offender_id:
                action_ind = np.hstack((action_ind, np.arange(5, 5 + ships)))
        else:
            action_ind = np.arange(9, 14)

        return action_ind

    # observation consists of:
    # 1. offender id, defender id, defender planet id
    # 2. each planet: owner id, ships on planet
    # 3. warp id, ships in the warp
    # 4. gate id, ships in the gate
    # 5. hand of player who turns now
    # ...
    # do we need to insert info about players' turns ?
    def __make_observation(self):
        observation = [self.offender_id, self.defender_id, self.defender_planet_id]
        for planet in self.planets:
            observation += [planet.owner_id] + planet.ships
        observation += [self.warp.owner_id] + self.warp.ships
        observation += [self.gate.owner_id] + self.gate.ships
        observation += self.player_hand[self.player_turns[0]]
        return np.array(observation, dtype=np.float_), self.negotiation_obs

    # responsible for:
    # 1. Reset Gate
    # 2. Determine defender
    # 3. Transfer defender's ships on Gate
    # 4. Determine players who can participate
    # 5. Regroup (compensate ship to offender)
    # 6. Fill hand up to CARDS_IN_HAND
    # 7. Negotiation observation
    def __reset_round(self):
        self.gate.offend_card = None
        self.gate.defend_card = None
        self.gate.player_side = [None] * self.nrof_players

        planets_to_offend = [i for i in range(len(self.planets)) if self.planets[i].owner_id != self.offender_id]
        self.defender_planet_id = np.random.choice(planets_to_offend)
        self.defender_id = self.planets[self.defender_planet_id].owner_id

        self.gate.ships[self.defender_id] = self.planets[self.defender_planet_id].ships[self.defender_id]
        self.planets[self.defender_planet_id].ships[self.defender_id] = 0
        self.gate.player_side[self.defender_id] = self.defender_id

        self.player_turns = [i for i in range(self.nrof_players)
            if i != self.offender_id and i != self.defender_id and self.planets[self.defender_planet_id].ships[i] == 0]
        if len(self.player_turns) > 0:
            next_id = self.offender_id + 1
            while next_id != self.offender_id:
                if next_id == self.defender_id:
                    next_id += 1
                if next_id == self.nrof_players:
                    next_id = 0
                if next_id in self.player_turns:
                    next_id = self.player_turns.index(next_id)
                    break
                next_id += 1
            self.player_turns = self.player_turns[next_id:] + self.player_turns[:next_id]
        self.player_turns = [self.offender_id] + self.player_turns + [self.offender_id, self.defender_id]

        self.__compensate(self.offender_id, 1)

        for player_id in range(self.nrof_players):
            if len(self.player_hand[player_id]) < self.Const.CARDS_IN_HAND:
                if len(self.deck) == 0:
                    np.random.shuffle(self.drop)
                    self.deck += self.drop
                    self.drop = []
                self.player_hand[player_id].append(self.deck.pop(0))

        self.negotiation_obs = torch.zeros((self.nrof_players, 10))

    # Takes into account:
    # 1. Morph card
    # 2. If both card are Negotiate - return ships on planets
    # 3. If one of side is Negotiate - another side win
    # 4. Whose attack power is greater, the side will win
    def __do_encounter(self):
        if self.gate.offend_card == self.Const.MORPH_CARD:
            self.gate.offend_card = self.gate.defend_card
        elif self.gate.defend_card == self.Const.MORPH_CARD:
            self.gate.defend_card = self.gate.offend_card

        if self.gate.offend_card == self.Const.NEGOTIATE_CARD and self.gate.defend_card == self.Const.NEGOTIATE_CARD:
            self.__from_gate_to_home(side=self.offender_id)
            self.__from_gate_to_home(side=self.defender_id)
        elif self.gate.defend_card == self.Const.NEGOTIATE_CARD:
            self.__from_gate_to_offended_planet(side=self.offender_id)
            self.__from_gate_to_warp(side=self.defender_id)
        elif self.gate.offend_card == self.Const.NEGOTIATE_CARD:
            for player_id in range(self.nrof_players):
                if self.gate.player_side[player_id] == self.defender_id:
                    self.__compensate(player_id, self.gate.ships[player_id])
            self.__from_gate_to_warp(side=self.offender_id)
            self.__from_gate_to_home(side=self.defender_id)
        else:
            offend_power = self.gate.offend_card
            defend_power = self.gate.defend_card
            for player_id in range(self.nrof_players):
                if self.gate.player_side[player_id] == self.offender_id:
                    offend_power += self.gate.ships[player_id]
                elif self.gate.player_side[player_id] == self.defender_id:
                    defend_power += self.gate.ships[player_id]
            if offend_power > defend_power:
                self.__from_gate_to_offended_planet(side=self.offender_id)
                self.__from_gate_to_warp(side=self.defender_id)
            else:
                for player_id in range(self.nrof_players):
                    if self.gate.player_side[player_id] == self.defender_id:
                        self.__compensate(player_id, self.gate.ships[player_id])
                self.__from_gate_to_warp(side=self.offender_id)
                self.__from_gate_to_home(side=self.defender_id)

        self.offender_id = (self.offender_id + 1) if (self.offender_id + 1) < self.nrof_players else 0

    # Game terminals if:
    # 1. A player achieve max number of not home planets
    # *. Reward shares uniformly between player which have the same number of offended planets
    def __check_win_condition(self):
        players = []
        terminal = False
        reward = 1.

        max_offended_planets = max(self.warp.planet_counter)
        if max_offended_planets == self.Const.PLANETS_TO_WIN:
            #print()
            print('Terminal state reached. Max number of planets was achieved by players!')
            #print(self.__str__(True, True, True, True, True))
            #print()
            terminal = True
            for player_id in range(self.nrof_players):
                if self.warp.planet_counter[player_id] == max_offended_planets:
                    players.append(player_id)

        reward = reward / len(players) if len(players) > 0 else reward
        return terminal, players, reward


    def __from_gate_to_warp(self, side):
        for player_id in range(self.nrof_players):
            if self.gate.player_side[player_id] == side:
                self.warp.ships[player_id] += self.gate.ships[player_id]
                self.gate.ships[player_id] = 0

    def __from_gate_to_home(self, side):
        for player_id in range(self.nrof_players):
            if self.gate.player_side[player_id] == side:
                for i in range(self.gate.ships[player_id]):
                    ships_on_planets = self.__get_ships_on_planets_by_player_id(player_id)
                    (planet_id, nrof_ships) = min(ships_on_planets, key=lambda x: x[1])
                    self.planets[planet_id].ships[player_id] += 1
                    self.gate.ships[player_id] -= 1

    def __from_gate_to_offended_planet(self, side):
        for player_id in range(self.nrof_players):
            if self.gate.player_side[player_id] == side:
                self.planets[self.defender_planet_id].ships[player_id] += self.gate.ships[player_id]
                self.gate.ships[player_id] = 0
                self.warp.planet_counter[player_id] += 1

    def __compensate(self, player_id, ships):
        for ship in range(ships):
            if self.warp.ships[player_id] != 0:
                self.warp.ships[player_id] -= 1
                ships_on_planets = self.__get_ships_on_planets_by_player_id(player_id)
                (planet_id, nrof_ships) = min(ships_on_planets, key=lambda x:x[1])
                self.planets[planet_id].ships[player_id] += 1

    def __get_ships_on_planets_by_player_id(self, player_id):
        player_planets = []
        for id, planet in enumerate(self.planets):
            if planet.ships[player_id] != 0 or planet.owner_id == player_id:
                player_planets.append((id, planet.ships[player_id]))
        return player_planets

    # Only debug information, presents game state
    def __str__(self, planet: bool = False, warp: bool = False, gate: bool = False, deck: bool = False, hand: bool = False):
        desc = '\nCosmic Encounter environment:\n\n'
        #desc = ''
        if planet:
            for id, planet in enumerate(self.planets):
                desc += 'planet ' + str(id) + ':\n' + str(planet) + '\n'
        if warp:
            desc += 'warp:\n' + str(self.warp) + '\n'
        if gate:
            desc += 'gate:\n' + str(self.gate) + '\n'
        if deck:
            desc += 'deck: ' + str(self.deck) + '\n'
        if hand:
            desc += 'hands:\n'
            for id, hand in enumerate(self.player_hand):
                desc += '    player ' + str(id) + ' hand: ' + str(hand) + '\n'
        return desc
