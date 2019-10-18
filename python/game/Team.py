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
    def closest_player_position(self, enemy_player_locations, x, y):
        player_position = [x, y]
        shortest_distance = float("inf")
        for key, value in enemy_player_locations.items():
            enemy_position = [value[0], value[1]]
            distance = sum(abs(a-b) for a,b in zip(player_position,enemy_position))
            if distance < shortest_distance:
                closest_enemy_location = value
                shortest_distance = distance
        return closest_enemy_location[0], closest_enemy_location[1]
        
        # Get current player's team_id
        #return x,y
