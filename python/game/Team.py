from typing import List, Dict
from game.Player import Player


class Team:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.resources: List = []

    def add_player(self, player: Player):
        self.players[player.id] = player
