import os

import numpy as np
import pygame
from model.ANN import ANN
from game.Game import Game

MAP_NAME = '10x10-ourgame.json'
NUM_OF_GAMES = 3
TRAIN = True
SKIP_RATE = 10


def play(g: Game, ann: ANN, game_num):
    # Initial 2 players

    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player3 = g.get_players(2)[1]
    player4 = g.get_players(2)[2]
    player1.main_player = True

    # Start the game (flag)
    g.start()
    g.set_train(True)
    g.update()
    g.update_state()
    g.prev_stat = g.get_state_stat()
    # print(g.get_resource_matrix())

    while True:
        # If the game is in terminal state
        if g.is_game_terminal() or g.get_ticks() > 3000:
            file = open("./logs_ann/evaluation.txt",'a+')
            g.game_result(file,game_num)
            file.close()
            g.stop()
            print("Game over")
            break
        
        # print("Player 1 : " + str(player1.location))
        # Player 1 action by model
        # TODO 599: Use the predicted action
        state = g.get_state()
        predicted_action = ann.predict_action(state)
        random_action = np.random.randint(2)
        if(np.random.random() < max(0.01, np.power(ann.epsilon, ann.tick))):
            action = random_action
        else:
            action = predicted_action

        #print("Player1 Action:", action)
        player1.do_action(action)
        # # # Player 2 random action
        player2.do_action(2)
        player3.do_action(2)
        player4.do_action(2)
        update_with_skip_rate(g, SKIP_RATE)
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        # print("Resource at ", i)
        # # print(g.get_resource_matrix())
        # print("Player Resource:", player1.player.getScore())
        g.view()  # View the game state in the pygame window

        next_state = g.get_state()
        reward = g.get_reward_value()

        ann.train(state, action, reward, next_state, g.is_terminal())

        state = next_state

def update_with_skip_rate(g, skip_rate):
    skip_count = 0
    while True:
        if g.update():
            if not TRAIN:
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
    if TRAIN:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        SKIP_RATE = 2
    
    file = open("./logs_ann/evaluation.txt",'a+')
    file.write("game_num,team_health,team_count,team_avg_health,team_resource,o_team_health,o_team_count,o_team_health_avg,o_team_resource,result,time_ticks\n")
    file.close()
    game = Game(MAP_NAME, train=TRAIN)
    game.add_a_player(2)
    game.add_a_player(2)
    ann = ANN()
    print(ann.get_summary())
    for _ in range(1):
        print("Game"+str(_+1)+" started.")
        play(game,ann,(_+1))
        game.reset()
        if(_ % 50 == 0):
            ann.save_model(str(int(_/50)))

