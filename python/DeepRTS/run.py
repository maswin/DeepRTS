import numpy
import pygame

from game.Game import Game

MAP_NAME = '31x31-6v6.json'
NUM_OF_GAMES = 3


def play(g: Game):
    pygame.init()

    # Initial 2 players
    player1 = g.players[0]
    player2 = g.players[1]

    # Start the game (flag)
    g.start()

    # Run forever
    while True:
        g.tick()  # Update the game clock
        g.update()  # Process the game state
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption

        g.view()  # View the game state in the pygame window

        events_hack()

        g.capture()

        # If the game is in terminal state
        if g.is_terminal():
            g.stop()
            print("Game over")
            break

        # Perform random action for player 1
        player1.do_action(get_random_action())

        # Perform random action for player 2
        player2.do_action(get_random_action())


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()


def get_random_action():
    return numpy.random.randint(0, 16)


if __name__ == "__main__":
    game = Game(MAP_NAME)
    game.add_a_player(1)
    game.add_a_player(2)
    for _ in range(NUM_OF_GAMES):
        play(game)
        game.reset()
