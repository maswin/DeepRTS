from game.Game import Game
from game.Player import Player
from math import sqrt
from game.util.constants import ATTACK_CLOSEST_TARGET, HARVEST_CLOSEST_RESOURCE, ATTACK_BASE


class RuleBasedAI:

    @staticmethod
    def distance(p1, p2):
        return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

    def predict(self, game: Game, player: Player):
        actions = []
        e_x, e_y = game.get_closest_enemy_location(player.location[0], player.location[1], player.team_id)
        actions.append((ATTACK_CLOSEST_TARGET, self.distance((e_x, e_y), player.location)))
        r_x, r_y = game.get_nearest_resource_index(player.location[0], player.location[1])
        actions.append((HARVEST_CLOSEST_RESOURCE, self.distance((r_x, r_y), player.location)))
        b_x, b_y = game.get_enemy_base(player.team_id)
        actions.append((ATTACK_BASE, self.distance((b_x, b_y), player.location)))
        return min(actions, key=lambda x: x[1])[0]
