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

    def __init__(self, map_name, config=None):
        super(Game, self).__init__(map_name, config=config)
        self.default_setup()

        # Initialize 2 teams with one player each
        self.teams: Dict[int, Team] = {1: Team(), 2: Team()}
        self.teams[1].add_player(Player(self.players[0], 1, self))
        self.teams[2].add_player(Player(self.players[1], 2, self))
        self.euclidean_mat = dict()
        self.manhattan_mat = dict()
        self.create_distance_matrices(10, 10);
        self.prev_stat = None;

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
        h = self.get_height()
        w = self.get_width()
        team_matrix = np.zeros((h, w))
        for k in team.players.keys():
            x,y = team.players[k].location
            team_matrix[y,x] = team.players[k].health_p
        return team_matrix

    def get_resource_matrix(self):
        tile_map = self.tilemap
        h = self.get_height()
        w = self.get_width()
        resource_matrix = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                tile_x_y: pyDeepRTS.Tile = tile_map.get_tile(i,j)
                if tile_x_y.is_harvestable:
                    resource_matrix[j,i] = tile_x_y.get_resources()
        
        return resource_matrix
    
    def get_state(self):
        our_team_matrix = self.get_team_matrix(self.teams[1])
        opponent_team_matrix = self.get_team_matrix(self.teams[2])
        resource_matrix = self.get_resource_matrix()
        state = np.stack([our_team_matrix,opponent_team_matrix,resource_matrix], axis = 2)
        print(state.shape)
        return state

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
        enemy_player_locations = dict()
        enemy_team = self.teams[Game.OPPONENTS[1]]
        for k in enemy_team.players.keys():
            enemy_player_locations[k] = enemy_team.players[k].location
        return enemy_team.closest_player_position(enemy_player_locations, x, y)

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
                if(inp[i][j] < mini) and not ((position.x,position.y) == (i,j)):
                    mini = inp[i][j]
                    min_i = i
                    min_j = j

        return (min_i-position.x,min_j-position.y);

    # Function to dismantle the abstract action to return list of atomic actions.

    def get_action_sequence(self, p1: Position, p2: Position) -> List[str]:

        # Converting column indexed co-ordinates to row indexed co-ordinates.
        p1_r = Position(p1.y, p1.x)
        p2_r = Position(p2.y, p2.x)
        # The manhattan distance matrix is transposed to convert to column indexed matrix from row indexed.
        grid = self.get_distance_matrices(p2_r.x, p2_r.y)[1]
        h,w = grid.shape 
        if((not p1_r.check_out_of_bounds(h,w)) or (not p2_r.check_out_of_bounds(h,w))):
            print(" Index not in bounds. ")
            return []
		
        action_sequence = []
        p = Position(p1_r.x, p1_r.y)
        #look at 8 neighbours find the direction with least distance.
        while(not p.is_equals(p2_r)):
            x, y  = self.get_min_neighbour(p, grid)
            action_sequence.append(ACTION_MAP_ROW_INDXD[(x,y)])
            p.x = p.x + x
            p.y = p.y + y
	
        return action_sequence


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
        return self.euclidean_mat[pos_x][pos_y] , self.manhattan_mat[pos_x][pos_y]

    
    def get_state_stat(self):

        team  = self.teams[1]
        ph_total = 0
        for p in team.players:
            if team.players[p].main_player:
                m_p = team.players[p];
                p_x,p_y = team.players[p].location
            ph_total += team.players[p].health_p

        o_team  = self.teams[2]
        o_count = 0;
        oh_total = 0
        for o_p in o_team.players:
            oh_total += o_team.players[o_p].health_p
            o_count += 1

        d_matrix = self.get_distance_matrices(p_y,p_x)[1]
        r_matrix = self.get_resource_matrix()
        nearest_resource_distance = np.amin((r_matrix > 0)*d_matrix)
        # Leaving the enemy distance for now.
        #nearest_enemy_distance = d_matrix[self.get_closest_enemy_location(p_x,p_y,1)];

        base_distance = d_matrix[1,1]

        mf = base_distance - nearest_resource_distance
        r_total = m_p.gold;

        return (mf,r_total, ph_total, (oh_total/o_count))


    def get_reward_value(self):

        self.curr_stat = self.get_state_stat();
        move_forward_diff = 0.5*10**(-5)*(self.curr_stat[0] - self.prev_stat[0])
        resource_difference = 0.001*(self.curr_stat[1] - self.prev_stat[1])
        health_difference = 0.5*(self.curr_stat[2] - self.prev_stat[2])
        opponent_health = self.curr_stat[3] - self.prev_stat[3]

        reward = move_forward_diff + resource_difference + health_difference - opponent_health

        print("Prev:", self.prev_stat, "Curr: ", self.curr_stat);
        self.prev_stat = self.curr_stat;
        return reward
    

