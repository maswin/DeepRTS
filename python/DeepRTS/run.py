import numpy
import pygame

from DeepRTS import PyDeepRTS

MAP_NAME = '10x10-2v2.json'
AUTO_ATTACK = True
AUTO_HARVEST = True
NUM_OF_GAMES = 3


def setup_game():
    g = PyDeepRTS(MAP_NAME)

    # Set FPS and UPS limits
    g.set_max_fps(10000000)
    g.set_max_ups(10000000)

    # How often the state should be drawn
    g.render_every(1)

    # How often the capture function should return a state
    g.capture_every(1)

    # How often the game image should be drawn to the screen
    g.view_every(1)

    # Configuration
    g.config.set_auto_attack(AUTO_ATTACK)
    g.config.set_harvest_forever(AUTO_HARVEST)

    return g


def play(g: PyDeepRTS):
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
    game = setup_game()
    for _ in range(NUM_OF_GAMES):
        play(game)
        game.reset()
