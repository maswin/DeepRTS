from typing import Tuple, List
import pyDeepRTS
from collections import deque
from game.util.constants import HISTORY_SIZE


class Player:

    def __init__(self, player: pyDeepRTS.Player):
        self.id: str = player.get_id()
        self.player: pyDeepRTS.Player = player
        self.location: Tuple = (0, 0)
        self.actions: List[int] = deque(maxlen=HISTORY_SIZE)

    def do_action(self, action_id):
        self.actions.append(action_id)
        self.player.do_action(action_id)
