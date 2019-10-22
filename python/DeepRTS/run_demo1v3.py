import numpy
import pygame

from game.Game import Game
from game.util.constants import RANDOM_MOVE, ATTACK_CLOSEST_TARGET, HARVEST_CLOSEST_RESOURCE
import collections
import time

MAP_NAME = '10x10-ourgame-1v3.json'
NUM_OF_GAMES = 3
TRAIN = False

player1_actions = [lambda: ATTACK_CLOSEST_TARGET] * 20 + [lambda: HARVEST_CLOSEST_RESOURCE] * 10

player2_actions = [lambda: RANDOM_MOVE]
player2_1_actions = [lambda: get_random_action()]
player2_2_actions = [lambda: get_random_action()]


def play(g: Game):
    # Initial 2 Teams
    player1 = g.get_players(1)[0]
    player1.main_player = True

    player2 = g.get_players(2)[0]
    player2_1 = g.add_a_player(2)
    player2_2 = g.add_a_player(2)

    start_game(g)

    for i in range(300):
        if g.is_game_terminal():
            g.stop()
            g.game_result()
            print("Game over")
            break

        # Spots
        print("Player 1    : " + str(player1.location) + " - " + str(player1.health_p) + " - " + str(player1.gold))
        print("Player 2    : " + str(player2.location) + " - " + str(player2.health_p) + " - " + str(player2.gold))
        print(
            "Player 2 (1): " + str(player2_1.location) + " - " + str(player2_1.health_p) + " - " + str(player2_1.gold))
        print(
            "Player 2 (2): " + str(player2_2.location) + " - " + str(player2_2.health_p) + " - " + str(player2_2.gold))

        player1.do_action(player1_actions[i % len(player1_actions)]())

        player2.do_action(player2_actions[i % len(player2_actions)]())
        player2_1.do_action(player2_1_actions[i % len(player2_1_actions)]())
        player2_2.do_action(player2_2_actions[i % len(player2_2_actions)]())

        time.sleep(0.05)

        g.update()
        g.update()
        g.tick()
        g.render()
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        g.view()  # View the game state in the pygame window
        events_hack()


def start_game(g):
    g.start()
    g.update()
    events_hack()
    g.update_state()


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()


def get_random_action():
    return numpy.random.randint(0, 16)


if __name__ == "__main__":
    for i in range(NUM_OF_GAMES):
        game = Game(MAP_NAME, train=TRAIN)
        play(game)
        game.reset()
