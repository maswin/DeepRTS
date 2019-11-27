# pybind config
RENDER_RATE = 1
CAPTURE_RATE = 1
VIEW_RATE = 1
MAX_FPS = 100000000
MAX_UPS = 100000000

# Game config
AUTO_ATTACK = False
AUTO_HARVEST = True

# Player config
HISTORY_SIZE = 10  # Number of actions to store

# Action ids
ATTACK_CLOSEST_TARGET = 0
HARVEST_CLOSEST_RESOURCE = 1
ATTACK_BASE = 2
DEFEND_BASE = 3
RANDOM_MOVE = 4
SELF_ATTACK = 5
NO_ACTION = 6

# Actual action
ATTACK = 18
HARVEST = 19

SKIP_RATE = 3
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
