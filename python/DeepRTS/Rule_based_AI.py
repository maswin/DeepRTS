from game.Game import Game
from game.util.constants import SKIP_RATE
import numpy as np
import pygame
import random
import json

MAP_NAME = '10x10-ourgame_gen.json'
NUM_OF_GAMES = 1

def map_generator():
    for a in range(30):
        matrix = []
        p1 = random.randint(0,9)
        p2 = random.randint(0,7)
        start_p1 = [p1+1,(9-p1)+1]
        start_p2 = [p2+1,p2+1]
        start_p3 = [p2+2, p2+2]
        for i in range(12):
            flag = True
            if i == 0 or i == 11:
                for j in range(12):
                    matrix.append(52)
                continue
            for j in range(12):
                if j == 0 or j == 11:
                    matrix.append(52)
                elif (i == start_p1[0] and j == start_p1[1]) or (i == start_p2[0] and j == start_p2[1]) or (i == start_p3[0] and j == start_p3[1]) or (i == start_p3[0]+1 and j == start_p3[1]+1):
                    matrix.append(17)
                else:
                    c = random.randint(0,99)
                    if c%5 == 0:
                        if flag and (matrix[12*(i-1) + j] != 102 or matrix[12*(i-1) + j + 1] != 102 or matrix[12*(i-1)+j-1] != 102):
                            matrix.append(102)
                            flag = False

                        else:
                            matrix.append(270)
                    else:
                        matrix.append(270)

        with open("assets/maps/10x10-ourgame.json", "r") as f:
            map_json = json.load(f)


        map_json_layer = map_json["layers"][0]
        map_json_layer["data"] = matrix
        map_json["layers"] = [map_json_layer]

        with open("assets/maps/10x10-ourgame_gen_"+str(a)+".json", "w+") as f:
            json.dump(map_json, f)
            f.close()
    exit()



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
            #print(player_object.location[0], player_object.location[1])
            #print(g.get_nearest_resource_index(player_object.location[0], player_object.location[1]))
            
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
    map_generator()
    game = Game(MAP_NAME)
    game.add_a_player(2)
    game.add_a_player(2)
    for _ in range(NUM_OF_GAMES):
        play(game)
        game.reset()

if __name__ == "__main__":
    main()
    


