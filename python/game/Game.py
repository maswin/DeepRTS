from game.Team import Team
from game.Player import Player
from game.util.constants import *
import pyDeepRTS
from pyDeepRTS import Unit
from DeepRTS import PyDeepRTS
from typing import Dict, List
import numpy as np


class Game(PyDeepRTS):
    OPPONENTS = {
        1: 2,
        2: 1
    }

    def __init__(self, map_name):
        super(Game, self).__init__(map_name)
        self.default_setup()

        # Initialize 2 teams with one player each
        self.teams: Dict[int, Team] = {1: Team(), 2: Team()}
        self.teams[1].add_player(Player(self.players[0], 1, self))
        self.teams[2].add_player(Player(self.players[1], 2, self))
        self.euclidean_mat = dict()
        self.manhattan_mat = dict()

    def default_setup(self):
        # Set FPS and UPS limits
        self.set_max_fps(MAX_FPS)
        self.set_max_ups(MAX_UPS)

        # How often the state should be drawn
        self.render_every(RENDER_RATE)

        # How often the capture function should return a state
        self.capture_every(CAPTURE_RATE)

        # How often the game image should be drawn to the screen
        self.view_every(VIEW_RATE)

        # Configuration
        self.config.set_auto_attack(AUTO_ATTACK)
        self.config.set_harvest_forever(AUTO_HARVEST)

    def get_players(self, team_id) -> List[Player]:
        return list(self.teams[team_id].players.values())

    def add_a_player(self, team_id):
        player: pyDeepRTS.Player = self.add_player()
        self.teams[team_id].add_player(Player(player, team_id, self))

    def _get_team(self, unit: Unit):
        for team in self.teams.values():
            if team.is_this_team(unit.get_player().get_id()):
                return team
        return None

    def update_state(self):
        units: List[Unit] = self.units
        for unit in units:
            team = self._get_team(unit)
            team.update(unit)

    def get_closest_enemy_location(self, x, y, team_id):
        enemy_team = self.teams[Game.OPPONENTS[team_id]]
        return enemy_team.closest_player_position(x, y)

    def _get_team_health(self, team_id):
        return [x.health_p for x in self.teams[team_id].players.values()]

    # TODO: Return a matrix to feed model
    def get_state_matrix(self):
        return [
            self._get_team_health(1),
            self._get_team_health(2)
        ]



    # TODO 599: Move to DQN or appropriate place



    def create_distance_matrices(self,length_of_map_x, height_of_map_y):
        def create_grid(x, y):
            x, y = np.mgrid[0:x:1, 0:y:1]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x;pos[:, :, 1] = y
            pos = np.reshape(pos, (x.shape[0], x.shape[1], 2))
            return pos
        grid = create_grid(length_of_map_x,height_of_map_y)
        for i in range(length_of_map_x):
            self.euclidean_mat[i] = dict()
            self.manhattan_mat[i] = dict()
            for j in range(height_of_map_y):
                custom_grid = grid-[i,j]
                self.euclidean_mat[i][j] = np.linalg.norm(custom_grid,axis =2)
                self.manhattan_mat[i][j] = np.sum(np.abs(custom_grid), axis=2)

    def get_distance_matrices(self,pos_x,pos_y):
        return self.euclidean_mat[pos_x,pos_y] , self.manhattan_mat[pos_x,pos_y]

