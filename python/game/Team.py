from typing import List, Dict
from game.Player import Player
from pyDeepRTS import Unit


class Team:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        # Not worrying about buildings for now
        # self.buildings: List = []

    def add_player(self, player: Player):
        self.players[player.id] = player

    def is_this_team(self, player_id):
        return player_id in self.players

    def update(self, unit: Unit):
        unit_tpe_id = int(unit.type_id)
        if unit_tpe_id == 1:
            self.update_player(unit)
        else:
            # Not worrying about buildings for now
            # self.update_buildings(unit)
            pass

    def update_player(self, unit):
        player = unit.get_player()
        self.players[player.get_id()].update(unit)

    # Find closest enemy location
    def closest_player_position(self, x, y):
        return 0, 0
