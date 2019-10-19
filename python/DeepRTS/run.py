import numpy
import pygame
from game.Game import Game
from game.util.constants import ATTACK, HARVEST, SKIP_RATE
import time


MAP_NAME = '10x10-2v2.json'

NUM_OF_GAMES = 3


def play(g: Game):
    pygame.init()

    # Initial 2 players
    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player1.main_player = True
    # Start the game (flag)
    g.start()

    x = 12
    y = 12

    g.prev_stat = g.get_state_stat();
    
    g.set_train(True)
    g.update()
    g.update_state()
    print("Reward value", g.get_reward_value())
    player1.player.do_my_action(HARVEST, 9, 5)
    update_with_skip_rate(g,13)
    g.update_state()
    print("Reward value", g.get_reward_value());
    player1.player.do_my_action(HARVEST, 9, 5)
    update_with_skip_rate(g,13)
    g.update_state()
    print("Reward value", g.get_reward_value());

    g.render()  # Draw the game state to graphics
    g.caption()  # Show Window caption
    
    # Run forever
    l = True
    while True:
        g.tick()  # Update the game clock
        update_with_skip_rate(g, SKIP_RATE)
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model

        g.view()  # View the game state in the pygame window

        b = g.capture()
        if l is True:
            numpy.save("tempoutfile.npy",b)
            l = False


        # If the game is in terminal state
        if g.is_terminal():
            g.stop()
            print("Game over")
            break


def update_with_skip_rate(g, skip_rate):
    skip_count = 0
    while True:
        if g.update():
            events_hack()
            skip_count += 1
        if skip_count == skip_rate:
            break


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()


def get_random_action():
    return numpy.random.randint(0, 16)


if __name__ == "__main__":
    game = Game(MAP_NAME)
    for _ in range(NUM_OF_GAMES):
        play(game)
        game.reset()
