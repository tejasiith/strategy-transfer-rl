import random

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SOURCE_MAP = [
    "#########",
    "#S..#..G#",
    "#.K.#...#",
    "#...D...#",
    "#C..#...#",
    "#########",
]
SOURCE_LEGEND = {
    "K": ("key", "tool"),
    "D": ("door", "barrier"),
    "G": ("goal", "objective"),
    "C": ("coin", "distractor"),
}

TARGET_MAP = [
    "###########",
    "#S....#...#",
    "#.##..#.#.#",
    "#.#L..T.#E#",
    "#.#...#.#.#",
    "#M....#...#",
    "###########",
]
TARGET_LEGEND = {
    "L": ("lever", "tool"),
    "T": ("gate", "barrier"),
    "E": ("exit", "objective"),
    "M": ("gem", "distractor"),
}

ONTOLOGY = {
    "key": "tool", "lever": "tool",
    "door": "barrier", "gate": "barrier",
    "goal": "objective", "exit": "objective",
    "coin": "distractor", "gem": "distractor",
}


class GridWorld:
    def __init__(self, layout, legend, max_steps, random_start=False, start_ok=None):
        self.grid = [list(row) for row in layout]
        self.height = len(layout)
        self.width = len(layout[0])
        self.max_steps = max_steps
        self.random_start = random_start
        self.objects = {}
        self.open_cells = []

        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                if ch in legend:
                    self.objects[(r, c)] = legend[ch]
                elif ch != "#":
                    self.open_cells.append((r, c))

        if start_ok is None:
            self.start_cells = self.open_cells
        else:
            self.start_cells = [p for p in self.open_cells if start_ok(p)]

    def reset(self, rng=random):
        if self.random_start:
            self.pos = rng.choice(self.start_cells)
        else:
            self.pos = self.start
        self.has_tool = False
        self.opened = False
        self.taken = set()
        self.steps = 0
        return self.state()

    def state(self):
        return (self.pos[0], self.pos[1], int(self.has_tool), int(self.opened))

    def step(self, action):
        self.steps += 1
        dr, dc = MOVES[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        event = None
        reward = 0.0
        done = False

        if self.grid[r][c] != "#":
            obj = self.objects.get((r, c))
            if obj and obj[1] == "barrier" and not self.opened:
                if self.has_tool:
                    self.opened = True
                    self.pos = (r, c)
                    event = ("unlock", obj[0])
            else:
                self.pos = (r, c)
                if obj and (r, c) not in self.taken:
                    name, role = obj
                    if role == "tool" or role == "distractor":
                        self.taken.add((r, c))
                        if role == "tool":
                            self.has_tool = True
                        event = ("acquire", name)
                    elif role == "objective":
                        event = ("reach", name)
                        reward = 1.0
                        done = True

        if self.steps >= self.max_steps:
            done = True
        return self.state(), reward, done, event


def make_source(random_start=False):
    # random starts stay in the left room, on the key side of the door
    return GridWorld(SOURCE_MAP, SOURCE_LEGEND, 200, random_start,
                     start_ok=lambda p: p[1] < 4)


def make_target():
    return GridWorld(TARGET_MAP, TARGET_LEGEND, 400)

HARD_MAP = [
    "#####################",
    "#S.........#........#",
    "#..........#........#",
    "#..........#........#",
    "#..........#........#",
    "#..........T........#",
    "#..........#........#",
    "#..........#........#",
    "#.......L..#........#",
    "#..........#M......E#",
    "#####################",
]


def make_hard_target():
    return GridWorld(HARD_MAP, TARGET_LEGEND, 500)