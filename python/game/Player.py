from typing import Tuple
import pyDeepRTS
from collections import deque
from game.util.constants import HISTORY_SIZE


class Player:

    def __init__(self, player: pyDeepRTS.Player):
        self.id: str = player.get_id()
        self.player: pyDeepRTS.Player = player
        self.location: Tuple = (0, 0)
        self.actions = deque(maxlen=HISTORY_SIZE)
        self.gold = 0
        self.damage_done = 0
        self.damage_taken = 0
        self.unit: pyDeepRTS.Unit = None

    def do_action(self, action_id):
        self.actions.append(action_id)
        self.player.do_action(action_id)

    def update(self, unit: pyDeepRTS.Unit):
        self.player = unit.get_player()
        self.unit = unit
        self.gold = self.player.sGatheredGold
        self.damage_done = self.player.sDamageDone
        self.damage_taken = self.player.sDamageTaken
        if unit.tile:
            self.location = (unit.tile.x, unit.tile.y)
