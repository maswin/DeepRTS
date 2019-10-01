import numpy
import pygame

from DeepRTS import PyDeepRTS

# Start the game
g = PyDeepRTS('10x10-2v2.json')

# Add players - By default 2 players gets added
# player1 = g.add_player()
# player2 = g.add_player()
player1 = g.players[0]
player2 = g.players[1]

print("Name : " + player1.get_name())
print("Name : " + player2.get_name())

# Set FPS and UPS limits
g.set_max_fps(10000000)
g.set_max_ups(10000000)

# How often the state should be drawn
g.render_every(1)

# How often the capture function should return a state
g.capture_every(1)

# How often the game image should be drawn to the screen
g.view_every(1)

# Start the game (flag)
g.start()

# Run forever
while True:
    g.tick()  # Update the game clock
    g.update()  # Process the game state
    g.render()  # Draw the game state to graphics
    # Captures current state (Returns None if .capture_every is set for some iterations)
    state = g.capture()
    g.caption()  # Show Window caption

    g.view()  # View the game state in the pygame window

    # Hack to render the pygame
    events = pygame.event.get()

    g.capture()

    # If the game is in terminal state
    if g.is_terminal():
        g.reset()  # Reset the game

    # Perform random action for player 1
    player1.do_action(numpy.random.randint(0, 16))

    # Perform random action for player 2
    player2.do_action(numpy.random.randint(0, 16))
