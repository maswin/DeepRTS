from DeepRTS import PyDeepRTS
import pygame
import numpy
g = PyDeepRTS('10x10-2v2.json')
player1 = g.players[0];
player2 = g.players[1];

g.set_max_fps(15)
g.set_max_ups(15)

g.render_every(1)
g.capture_every(1)
g.view_every(1)

g.start()
#player2.do_my_action(16, -1, -1)
#player1.do_my_action(17, 6, 7)
player1.do_my_action(19,0,0);

# Run forever
while True:
    g.tick()  # Update the game clock
    g.update()  # Process the game state
    g.render()  # Draw the game state to graphics
    g.caption()  # Show Window caption
#   g.update_state()  # Update states to new model

    g.view()  # View the game state in the pygame window

    events = pygame.event.get()
    g.capture()

    print(player1.getScore())
    # If the game is in terminal state
    if g.is_terminal():
        g.stop()
        print("Game over")
        break

        # Perform random action for player 1
        # x = numpy.random.randint(3, 20)
        # y = numpy.random.randint(3, 20)
        # player1.move_to(x, y)
        #player1.do_my_action(11,-1,-1)

        # Perform random action for player 2
        # x = numpy.random.randint(3, 20)
        # y = numpy.random.randint(3, 20)


