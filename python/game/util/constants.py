# pybind config
RENDER_RATE = 1
CAPTURE_RATE = 1
VIEW_RATE = 1
MAX_FPS = 10000000
MAX_UPS = 10000000

# Game config
AUTO_ATTACK = False
AUTO_HARVEST = True

# Player config
HISTORY_SIZE = 10  # Number of actions to store

# Action ids
ATTACK_CLOSEST_TARGET = 1
HARVEST_CLOSEST_RESOURCE = 2
RANDOM_MOVE = 3
NO_ACTION = 4

# Action map for row indexed matrix.
ACTION_MAP_ROW_INDXD = {
(-1, 0): "Up",
(1, 0): "Down",
(0, -1): "Left",
(0, 1): "Right",
(-1, -1): "UpLeft",
(1, -1): "DownLeft",
(1, 1): "DownRight",
(-1, 1): "UpRight"
}