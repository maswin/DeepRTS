import numpy
import pygame

from game.Game import Game
from game.util.constants import SKIP_RATE
from model.DDQN import DoubleDeepQNetwork
from pyDeepRTS import Config
import numpy as np

MAP_NAME = '10x10-2v2.json'
NUM_OF_GAMES = 3


def play(g: Game, ddqn: DoubleDeepQNetwork):
    pygame.init()

    # Initial 2 players
    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player1.main_player = True

    # Start the game (flag)
    g.start()
    g.set_train(True)

    # g.tick()  # Update the game clock
    # update_with_skip_rate(g, SKIP_RATE)
    # g.render()  # Draw the game state to graphics
    # g.caption()  # Show Window caption
    # g.update_state()  # Update states to new model
    # g.view()  # View the game state in the pygame window

    state = g.capture_grey_scale()

    while True:
        # If the game is in terminal state
        if g.is_terminal():
            g.stop()
            print("Game over")
            break

        print("Player 1 : " + str(player1.location))

        # Player 1 action by model
        # TODO 599: Use the predicted action
        action = ddqn.predict_action(state)
        player1.move_to(5, 5)

        # Player 2 random action
        player2.do_action(numpy.random.randint(1, 3))

        update_with_skip_rate(g, 10)
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        g.view()  # View the game state in the pygame window

        next_state = g.capture_grey_scale()
        # TODO 599: Use actual reward
        reward = 1

        ddqn.train(state, action, reward, next_state, g.is_terminal())

        state = next_state


def update_with_skip_rate(g, skip_rate):
    skip_count = 0
    while True:
        if g.update():
            events_hack()
            g.tick()
            skip_count += 1
        if skip_count == skip_rate:
            break


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()


def get_random_action():
    return numpy.random.randint(0, 16)


if __name__ == "__main__":
    # config = {"tickModifier": 1}
    # config = Config()
    # config.set_tick_modifier(1)
    game = Game(MAP_NAME)

    ddqn = DoubleDeepQNetwork()
    for _ in range(NUM_OF_GAMES):
        play(game, ddqn)
        game.reset()
    print(ddqn.get_summary())
