<p align="center">
  <img src ="./docs/logo.png" />
</p>


## Dependencies

* Python >= 3.x
* GCC >= 5.5

# Installation
```
git clone https://github.com/maswin/DeepRTS.git
cd DeepRTS
conda create -n deep-rts python=3.6
source activate deep-rts
git submodule sync
git submodule update --init --recursive
sudo pip install .
```
# Available maps
```
10x10-2-FFA
15x15-2-FFA
21x21-2-FFA
31x31-2-FFA
31x31-4-FFA
31x31-6-FFA
```

# Example
```python
import numpy

from pyDeepRTS import PyDeepRTS

# Start the game
g = PyDeepRTS()

# Add players
player1 = g.players[0]
player2 = g.players[1]

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
    state = g.capture()  # Captures current state (Returns None if .capture_every is set for some iterations)
    
    g.caption()  # Show Window caption
    # Hack to render the pygame
    events = pygame.event.get()
    g.view()  # View the game state in the pygame window
    
    # If the game is in terminal state
    if g.is_terminal():
        g.reset()  # Reset the game

    # Perform random action for player 1
    player1.do_action(numpy.random.randint(0, 16))
    
    # Perform random action for player 2
    player2.do_action(numpy.random.randint(0, 16))
```
