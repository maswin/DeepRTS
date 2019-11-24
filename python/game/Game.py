from game.Team import Team
from game.Player import Player
from game.util.constants import *
import pyDeepRTS
from pyDeepRTS import Unit
from DeepRTS import PyDeepRTS
from typing import Dict, List
import numpy as np
from game.Position import Position


class Game(PyDeepRTS):
    OPPONENTS = {
        1: 2,
        2: 1
    }

    def __init__(self, map_name, config=None, train=False):
        super(Game, self).__init__(map_name, config=config, train=train)
        self.default_setup()

        # Initialize 2 teams with one player each
        self.teams: Dict[int, Team] = {1: Team(), 2: Team()}
        self.teams[1].add_player(Player(self.players[0], 1, self))
        self.teams[2].add_player(Player(self.players[1], 2, self))
        self.euclidean_mat = dict()
        self.manhattan_mat = dict()
        self.create_distance_matrices(10, 10)
        self.prev_stat = None

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

    def get_team_matrix(self, team):
        # h = self.get_height()
        # w = self.get_width()
        h = self.get_height() - 2
        w = self.get_width() - 2
        team_matrix = np.zeros((h, w))
        for k in team.players.keys():
            x, y = team.players[k].location
            # team_matrix[y, x] = team.players[k].health_p
            team_matrix[y - 1, x - 1] = team.players[k].health_p
        return team_matrix

    def get_resource_matrix(self):
        tile_map = self.tilemap
        h = self.get_height()
        w = self.get_width()
        # resource_matrix = np.zeros((h, w))
        resource_matrix = np.zeros((h - 2, w - 2))
        # for i in range(1, h):
        for i in range(1, h - 1):
            # for j in range(1, w):
            for j in range(1, w - 1):
                tile_x_y: pyDeepRTS.Tile = tile_map.get_tile(i, j)
                if tile_x_y.is_harvestable:
                    # resource_matrix[j, i] = tile_x_y.get_resources()
                    resource_matrix[j - 1, i - 1] = tile_x_y.get_resources()
        return resource_matrix

    def is_unit_harvestable(self, i, j):
        tile_map = self.tilemap
        tile_x_y: pyDeepRTS.Tile = tile_map.get_tile(i, j)
        return tile_x_y.is_harvestable()

    def is_unit_attackable(self, i, j):
        tile_map = self.tilemap
        tile_x_y: pyDeepRTS.Tile = tile_map.get_tile(i, j)
        return tile_x_y.is_attackable()

    def get_state(self):
        our_team_matrix = self.get_team_matrix(self.teams[1])
        opponent_team_matrix = self.get_team_matrix(self.teams[2])
        resource_matrix = self.get_resource_matrix()
        state = np.stack([our_team_matrix, opponent_team_matrix, resource_matrix], axis=2)
        return state.reshape(1, 300)

    def add_a_player(self, team_id):
        player: pyDeepRTS.Player = self.add_player()
        player = Player(player, team_id, self)
        self.teams[team_id].add_player(player)
        return player

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
        enemy_player_locations = dict()
        enemy_team = self.teams[Game.OPPONENTS[team_id]]
        for k in enemy_team.players.keys():
            enemy_player = enemy_team.players[k]
            if not enemy_player.player.is_defeated():
                enemy_player_locations[k] = enemy_team.players[k].location
        return enemy_team.closest_player_position(enemy_player_locations, x, y)

    def get_enemy_base(self, team_id):
        return self.teams[Game.OPPONENTS[team_id]].base_location

    def get_enemy_locations(self, x, y, team_id):
        enemy_player_locations = dict()
        enemy_team = self.teams[Game.OPPONENTS[team_id]]
        for k in enemy_team.players.keys():
            enemy_player_locations[k] = enemy_team.players[k].location
        return enemy_player_locations

    def _get_team_health(self, team_id):
        return [x.health_p for x in self.teams[team_id].players.values()]

    # TODO: Return a matrix to feed model
    def get_state_matrix(self):
        return [
            self._get_team_health(1),
            self._get_team_health(2)
        ]

    # Function to return the co-ordinates which when added to current player location
    # puts it closer to destination.

    def get_min_neighbour(self, position, inp):
        rows, cols = inp.shape
        mini = 99999
        min_i = 0
        min_j = 0
        for i in range(max(0, position.x - 1), min(rows, position.x + 2)):
            for j in range(max(0, position.y - 1), min(cols, position.y + 2)):
                if (inp[i][j] < mini) and not ((position.x, position.y) == (i, j)):
                    mini = inp[i][j]
                    min_i = i
                    min_j = j

        return min_i - position.x, min_j - position.y

    # Function to dismantle the abstract action to return list of atomic actions.

    def get_action_sequence(self, p1: Position, p2: Position) -> List[str]:

        # Converting column indexed co-ordinates to row indexed co-ordinates.
        p1_r = Position(p1.y, p1.x)
        p2_r = Position(p2.y, p2.x)
        # The manhattan distance matrix is transposed to convert to column indexed matrix from row indexed.
        grid = self.get_distance_matrices(p2_r.x, p2_r.y)[1]
        h, w = grid.shape
        if (not p1_r.check_out_of_bounds(h, w)) or (not p2_r.check_out_of_bounds(h, w)):
            print(" Index not in bounds. ")
            return []

        action_sequence = []
        p = Position(p1_r.x, p1_r.y)
        # look at 8 neighbours find the direction with least distance.
        while not p.is_equals(p2_r):
            x, y = self.get_min_neighbour(p, grid)
            action_sequence.append(ACTION_MAP_ROW_INDXD[(x, y)])
            p.x = p.x + x
            p.y = p.y + y

        return action_sequence

    # TODO 599: Move to DQN or appropriate place

    def create_distance_matrices(self, length_of_map_x, height_of_map_y):
        def create_grid(x, y):
            x, y = np.mgrid[0:x:1, 0:y:1]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            pos = np.reshape(pos, (x.shape[0], x.shape[1], 2))
            return pos

        grid = create_grid(length_of_map_x, height_of_map_y)
        for i in range(length_of_map_x):
            self.euclidean_mat[i] = dict()
            self.manhattan_mat[i] = dict()
            for j in range(height_of_map_y):
                custom_grid = grid - [i, j]
                self.euclidean_mat[i][j] = np.linalg.norm(custom_grid, axis=2)
                self.manhattan_mat[i][j] = np.sum(np.abs(custom_grid), axis=2)

    def get_distance_matrices(self, pos_x, pos_y):
        return self.euclidean_mat[pos_x][pos_y], self.manhattan_mat[pos_x][pos_y]

    def get_nearest_resource_index(self, p_x, p_y):
        d_matrix = self.get_distance_matrices(p_y, p_x)[1]
        r_matrix = self.get_resource_matrix()
        near_res_mat = (r_matrix > 0) * d_matrix
        near_res_mat_non_zero = near_res_mat[np.nonzero(near_res_mat)]
        if len(near_res_mat_non_zero) == 0:
            return (-1, -1)
        min_v = min(near_res_mat_non_zero)
        near_res_ind = np.argwhere(near_res_mat == min_v)[0]
        # return (near_res_ind[1], near_res_ind[0])
        return near_res_ind[1] + 1, near_res_ind[0] + 1

    def get_state_stat(self):
        team = self.teams[1]
        ph_total = team.get_total_health()
        m_p = team.get_main_player()
        p_x, p_y = m_p.location

        o_team = self.teams[2]
        o_count = o_team.get_player_count()
        o_count = 1 if not o_count else o_count

        oh_total = o_team.get_total_health()

        d_matrix = self.get_distance_matrices(p_y - 1, p_x - 1)[1]
        r_matrix = self.get_resource_matrix()
        nearest_resource_distance = np.amin((r_matrix > 0) * d_matrix)
        # Leaving the enemy distance for now.
        # nearest_enemy_distance = d_matrix[self.get_closest_enemy_location(p_x,p_y,1)];

        base_distance = d_matrix[1, 1]

        mf = base_distance - nearest_resource_distance
        r_total = m_p.gold

        return mf, r_total, ph_total, (oh_total / o_count)

    def get_reward_value(self):
        curr_stat = self.get_state_stat()
        move_forward_diff = 0.5 * 10 ** (-5) * (curr_stat[0] - self.prev_stat[0])
        resource_difference = 0.001 * (curr_stat[1] - self.prev_stat[1])
        health_difference = 0.5 * (curr_stat[2] - self.prev_stat[2])
        opponent_health = curr_stat[3] - self.prev_stat[3]
        w_reward = 2 * self.won_or_lost()
        reward = move_forward_diff + resource_difference + health_difference - opponent_health + w_reward

        # print("Prev:", self.prev_stat, "Curr: ", curr_stat)
        self.prev_stat = curr_stat
        return reward

    def score(self, team):
        return (team.get_total_health() * 100) + (team.get_total_resources() * 5)

    def is_game_terminal(self):
        team = self.teams[1]
        team_health = team.get_total_health()
        o_team = self.teams[2]
        o_team_health = o_team.get_total_health()

        return team_health == 0.0 or o_team_health == 0.0 or self.get_ticks() > 4000

    def won_or_lost(self):
        if (not self.is_game_terminal):
            return 0
        team = self.teams[1]
        team_health = team.get_total_health()
        o_team = self.teams[2]
        o_team_health = o_team.get_total_health()
        return (1 if team_health >= o_team_health else -1)

    def game_result(self, f=None, game_num=0):
        team = self.teams[1]
        team_health = team.get_total_health()
        team_resource = team.get_total_resources()
        team_count = team.get_player_count()
        team_health_avg = (team_health / (team_count * 1.0))

        o_team = self.teams[2]
        o_team_health = o_team.get_total_health()
        o_team_resource = o_team.get_total_resources()
        o_team_count = o_team.get_player_count()
        o_team_health_avg = (o_team_health / (o_team_count * 1.0))

        result = "W" if team_health >= o_team_health else "L"

        time_taken = self.get_ticks()

        result = ",".join([str(game_num), str(team_health), str(team_count), str(team_health_avg), str(team_resource),
                           str(o_team_health), str(o_team_count), str(o_team_health_avg), str(o_team_resource),
                           str(result), str(time_taken)])

        if f:
            f.write(result + "\n")
        print("Game result:" + result)
        return result
