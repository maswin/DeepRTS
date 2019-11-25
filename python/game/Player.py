from typing import Tuple
import pyDeepRTS
from collections import deque
from game.util.constants import *
from game import Game
import numpy as np
import random


class Player:

    def __init__(self, player: pyDeepRTS.Player, team_id, game: Game):
        self.id: str = player.get_id()
        self.team_id = team_id
        self.player: pyDeepRTS.Player = player
        self.location: Tuple = (0, 0)
        self.actions = deque(maxlen=HISTORY_SIZE)

        # Added this attribute to find closest enemy and resource details
        self.game = game

        # Features : Add more if needed. Look for unit.cpp and player.cpp file in bindings package for the
        # attributes you have. Update it in update method
        self.health_p = 0
        self.gold = 0
        self.damage_done = 0
        self.damage_taken = 0
        self.main_player = False

    def build_town_hall(self):
        return self.player.do_my_action(14, -1, -1)

    def do_action(self, action_id):
        self.actions.append(action_id)
        if action_id == ATTACK_CLOSEST_TARGET:
            # TODO: Complete find closest enemy method
            x, y = self.game.get_closest_enemy_location(self.location[0], self.location[1], self.team_id)
            self.player.do_my_action(18, x, y)
        elif action_id == HARVEST_CLOSEST_RESOURCE:
            # TODO: now it will harvest either gold or lumber. Discuss if we should restrict it to gold
            x, y = self.game.get_nearest_resource_index(self.location[0]-1, self.location[1]-1)
            if(x == -1 and y == -1):
                self.move_to(np.random.randint(2, 10), np.random.randint(2, 10))
                return
            self.player.do_my_action(19, x, y)
        elif action_id == RANDOM_MOVE:
            # TODO: Make it random
            move = np.random.randint(3, 10)
            self.player.do_my_action(move, -1, -1)
        elif action_id == ATTACK_BASE:
            x, y = self.game.get_enemy_base(self.team_id)
            self.player.do_my_action(18, x, y)
        elif action_id == DEFEND_BASE:
            x,y = self.game.get_friend_base(self.team_id)
            i,j = self.game.get_closest_enemy_location(x, y, self.team_id)
            self.player.do_my_action(18,i,j)



        else:
            # Nothing to do
            pass

    def update(self, unit: pyDeepRTS.Unit):
        self.health_p = unit.health / unit.health_max
        self.gold = unit.gold_carry
        self.damage_done = unit.get_player().sDamageDone
        self.damage_taken = unit.get_player().sDamageTaken
        if unit.tile:
            self.location = (unit.tile.x, unit.tile.y)

    def move_to(self, x, y):
        self.player.do_my_action(17, x, y)

    def is_alive(self):
        return self.health_p > 0

