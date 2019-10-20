from game.Game import Game
from game.util.constants import SKIP_RATE
import numpy as np
import pygame
import random

MAP_NAME = '10x10-ourgame.json'
NUM_OF_GAMES = 1

def play(g: Game):
    pygame.init()
    #Initialize one player
    player1 = g.get_players(1)[0]
    #g.add_a_player(1)
    #player3 = g.get_players(1)[1]
    
    player2 = g.get_players(2)[0]
    #g.add_a_player(2)
    #player4 = g.get_players(2)[1]
    

    # Start the game (flag)
    g.start()
    while True:
        g.tick()  # Update the game clock
        g.update()  # Process the game state
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        
        for player_id, player_object in g.teams[1].players.items():
            x, y = player_object.location[0]+random.randint(-1,1), player_object.location[1]+random.randint(-1,1)
            player_object.move_to(x, y)
            possible_player_movements = [(x+0,y+1), (x+0, y-1), (x+1,y+0), (x-1, y+0), (x+1,y+1), (x-1, y-1), (x+1,y-1), (x-1, y+1)]
            harvestable_units = [i for i in possible_player_movements if  g.is_unit_harvestable(i[0],i[1])]
            enemy_locations = g.get_enemy_locations(x,y,1)
            attackable_units = [v for k,v in enemy_locations.items() if v in possible_player_movements]

            if len(attackable_units) != 0:
                v = random.choice(attackable_units)
                player_object.player.do_my_action(11, v[0], v[1])
            elif len(harvestable_units) != 0:
                v = random.choice(harvestable_units)
                player_object.player.do_my_action(11, v[0], v[1])
            else:
                continue

        for enemy_id, enemy_object in g.teams[2].players.items():
            enemy_object.do_action(action_id = 'RANDOM_MOVE')
        g.update_state()

        g.view()  # View the game state in the pygame window

        events_hack()

        g.capture()

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
            g.tick()
            skip_count += 1
        if skip_count == skip_rate:
            break


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()

def main():
    game = Game(MAP_NAME)
    for _ in range(NUM_OF_GAMES):
        play(game)
        game.reset()

if __name__ == "__main__":
    main()
    


