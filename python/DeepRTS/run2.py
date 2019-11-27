import os
import time
import numpy as np
import pygame
from model.ANN import ANN
from game.Game import Game

MAP_NAME = '13x13-ourgame_map1.json'
NUM_OF_GAMES = 100
TRAIN = True
SKIP_RATE = 15


def play(g: Game, ann: ANN, game_num):
    # Initial 2 players

    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player3 = g.get_players(2)[1]
    #player4 = g.get_players(2)[2]
    player1.main_player = True
    

    # Start the game (flag)
    g.start()
    g.set_train(True)
    g.update()
    g.update_state()
    player1.build_town_hall()
    player3.build_town_hall()
    update_with_skip_rate(g, 10)
    g.update_state()
    g.prev_stat = g.get_state_stat()
    # print(g.get_resource_matrix())
    state = g.get_state()

    i = 0
    while True:
        # If the game is in terminal state
        if not TRAIN:
            time.sleep(0.3)
        if g.is_game_terminal():
            if TRAIN:
                ex = "PER" if ann.PER else "ER"
                file = open("./logs_ann/evaluation_"+ex+".csv",'a+')
                g.game_result(file,game_num)
                file.close()
            else:
                g.game_result(None, game_num)

            g.stop()
            print("Game over")
            break
        
        # print("Player 1 : " + str(player1.location))
        # Player 1 action by model
        # TODO 599: Use the predicted action
        
        predicted_action = ann.predict_action(state)
        random_action = np.random.randint(4)
        if(np.random.random() < max(0.01, np.power(ann.epsilon, ann.tick))):
            action = random_action
        else:
            action = predicted_action

        #print("Player1 Action:", action)
        player1.do_action(action)
        
        # # # # Player 2 random action
        player2.do_action(random_action)
        player3.do_action(4)
        #player4.do_action(2)
        update_with_skip_rate(g, SKIP_RATE)

        g.update_state()  # Update states to new model
        g.view()  # View the game state in the pygame window

        next_state = g.get_state()
        reward = g.get_reward_value()

        ann.train(state, action, reward, next_state, g.is_game_terminal())

        state = next_state
        # i += 1

def update_with_skip_rate(g, skip_rate):
    skip_count = 0
    while True:
        if g.update():
            if not TRAIN:
                events_hack()
            g.tick()
            g.render()
            skip_count += 1
        if skip_count == skip_rate:
            break


def events_hack():
    # Hack to render the pygame
    events = pygame.event.get()


def get_random_action():
    return numpy.random.randint(0, 16)


if __name__ == "__main__":

    game = Game(MAP_NAME, train = TRAIN)
    game.add_a_player(2) 
    ann = ANN()
    ann.set_PER(False)
    #print(ann.get_summary())

    if TRAIN:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        SKIP_RATE = 2
        ex = "PER" if ann.PER else "ER"
        file = open("./logs_ann/evaluation_"+ ex +".csv",'a+')
        file.write("game_num,team_health,team_count,team_avg_health,team_resource,o_team_health,o_team_count,o_team_health_avg,o_team_resource,result,time_ticks\n")
        file.close()
    


    for _ in range(NUM_OF_GAMES):
        print("Game"+str(_+1)+" started.")
        play(game,ann,(_+1))
        if(TRAIN and _ % 100 == 0):
            ann.save_model(str(int(_/100)))
        game.reset()

