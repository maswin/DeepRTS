import numpy
import pygame

from game.Game import Game
from model.DDQN import DoubleDeepQNetwork
from pyDeepRTS import Config
import itertools as it
import collections
from model.NPC_CatBoost import NPC_CatBoost
from model.NPC_CatBoost import NPC_History
import numpy as np
import os

MAP_NAME = '10x10-2v2.json'
NUM_OF_GAMES = 3
TRAIN = True
SKIP_RATE = 10


def play(g: Game, ddqn: DoubleDeepQNetwork,NPC_Memory: NPC_History, use_NPC=False):
    # Initial 2 players

    player1 = g.get_players(1)[0]
    player2 = g.get_players(2)[0]
    player1.main_player = True
    if use_NPC is True:
        action_list = collections.deque(maxlen=5)

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
    g.prev_stat = g.get_state_stat()

    while True:
        # If the game is in terminal state
        if g.is_terminal():
            g.stop()
            action_list.clear()
            print("Game over")
            break

        print("Player 1 : " + str(player1.location))

        # Player 1 action by model
        # TODO 599: Use the predicted action
        action = ddqn.predict_action(state)


        # Only for NPC as current model is no good
        if use_NPC is True:
            npc_action = np.random.randint(2)

            if (len(action_list)>=5):
                # TODO dummy_get_npc_state
                player_health = player1.health_p
                resource_matrix = g.get_resource_matrix()
               # print(resource_matrix.shape)
                x, y = g.get_closest_enemy_location(player1.location[0], player1.location[1], 1)
                dist_closest_enemy = (np.abs(player1.location[0]-x) + np.abs(player2.location[1]-y))
                reward_value = g.get_reward_value()
                number_of_enemies = len(g.teams[Game.OPPONENTS[1]].players)
                NPC_state = [action_list[0],action_list[1],action_list[2],action_list[3],action_list[4],player_health,np.sum(np.sum(resource_matrix,axis=0),axis = 0),dist_closest_enemy,reward_value,number_of_enemies]
                #print(NPC_state)
                #NPC_Memory.Add_Observation(NPC_state,action)
                NPC_Memory.Add_Observation(NPC_state, npc_action)
                print(NPC_Memory.num_obs)
                if (NPC_Memory.num_obs > 100 and NPC_Memory.do_once_flag is True ):

                    print("GOT HERE __________----------")
                    NPC_History.do_once_flag = False
                    player_net = NPC_CatBoost('Follower')
                    player_net.train_(NPC_Memory.CAT_State, NPC_Memory.CAT_Action)
                    player_net.eval_train()
            if use_NPC is True:
                action_list.append(action)
        player1.do_action(action)

        # Player 2 random action
        player2.do_action(numpy.random.randint(1, 3))

        update_with_skip_rate(g, SKIP_RATE)
        g.render()  # Draw the game state to graphics
        g.caption()  # Show Window caption
        g.update_state()  # Update states to new model
        g.view()  # View the game state in the pygame window

        next_state = g.capture_grey_scale()
        reward = g.get_reward_value()

        ddqn.train(state, action, reward, next_state, g.is_terminal())

        state = next_state


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

    #game = Game(MAP_NAME, train=TRAIN)
    game = Game(MAP_NAME, train=TRAIN)
    NPC_Memory = NPC_History()
    ddqn = DoubleDeepQNetwork()

    for i in range(100):
        play(game, ddqn,NPC_Memory)
        if(i % 10 == 0):
            iteration = str(int(i/10))
            ddqn.save_model(iteration)
        game.reset()
    print(ddqn.get_summary())
