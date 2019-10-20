import os

import numpy
import pygame

from game.Game import Game

MAP_NAME = '10x10-2v2.json'
NUM_OF_GAMES = 3
TRAIN = True
SKIP_RATE = 10


def play(g: Game):
    # Initial 2 players

    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player1.main_player = True

    # Start the game (flag)
    g.start()
    g.set_train(True)

    while True:
        # If the game is in terminal state
        if g.is_terminal():
            g.stop()
            print("Game over")
            break

        print("Player 1 : " + str(player1.location))

        # Player 1 action by model
        # TODO 599: Use the predicted action
        # action = ddqn.predict_action(state)
        player1.move_to(8, 8)

        # Player 2 random action
        player2.do_action(numpy.random.randint(1, 3))

        update_with_skip_rate(g, SKIP_RATE)
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        g.view()  # View the game state in the pygame window


def update_with_skip_rate(g, skip_rate):
    skip_count = 0
    while True:
        if g.update():
            if not TRAIN:
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
    if TRAIN:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        SKIP_RATE = 1

    game = Game(MAP_NAME, train=TRAIN)

    for _ in range(45):
        play(game)
        game.reset()
    print(ddqn.get_summary())
