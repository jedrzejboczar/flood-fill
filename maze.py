import copy
import itertools
import subprocess
import time
from random import randrange, shuffle

import numpy as np


def clear_screen():
    subprocess.call(["clear"])
def sleep(sec):
    time.sleep(sec)

X = 8
Y = 8

# when they are 0 it disables printing corresponding steps
FLOODFILL_DELAY = 0.1
MOVE_DELAY = 0.1
INTERACTIVE = True
NO_PRINT_CELLS = False

MAX_STACK_SIZE = -1


class Cell():
    def __init__(self, N=0, E=0, S=0, W=0, weight=-1):
        self.weight = weight
        self.N = N != 0
        self.S = S != 0
        self.E = E != 0
        self.W = W != 0
        self.x = -1
        self.y = -1

    def wall(self, direction):
        assert direction in ['N', 'E', 'S', 'W']
        return getattr(self, direction)

    def setWall(self, direction, value):
        assert direction in ['N', 'E', 'S', 'W']
        setattr(self, direction, value)

    @staticmethod
    def fromListOfCells(cells):
        """Creates numpy 2D array of cells from a list of tuples.
        Each position in list is a tuple of values for this cell's __init__.
        The cells should be specified in 1D array that gets reshaped to 2D one."""
        cells = np.array(cells)
        cells = np.fliplr(cells.reshape((X, Y)).transpose())
        for x in range(X):
            for y in range(Y):
                cells[x, y].x = x
                cells[x, y].y = y
        return cells

    @staticmethod
    def fromList(listoftuples):
        cells = [Cell(*i) for i in listoftuples]
        return Cell.fromListOfCells(cells)

    @staticmethod
    def emptyMaze():
        return Cell.fromList([(0, 0, 0, 0) for _ in range(X * Y)])

    def __repr__(self):
        # return str(self.weight)
        return '<%d, %d>' % (self.x, self.y)


# prints the maze, if given shows mouse as @ and target as $
# (if mouse is in target, show mouse only)
def printCells(cells, current_pos=None, target_pos=None):
    if NO_PRINT_CELLS:
        return
    string = ""
    for y in range(Y - 1, -1, -1):
        for x in range(X):
            string += "+{}+".format("====" if cells[x][y].N else "    ")
        string += "\n"
        for x in range(X):  # od X-1 do 0
            fmt = "{}{:3} {}"
            if target_pos is not None:
                tx, ty = target_pos
                if tx == x and ty == y:
                    fmt = "{}{:3}${}"
            if current_pos is not None:
                cx, cy = current_pos
                if cx == x and cy == y:
                    fmt = "{}{:3}@{}"
            # weights
            string += fmt.format("|" if cells[x][y].W else " ",
                                 cells[x][y].weight,
                                 "|" if cells[x][y].E else " ")
        string += "\n"
        for x in range(X):
            string += "+{}+".format("====" if cells[x][y].S else "    ")
        string += "\n"
    print(string)


class Maze():
    # printing_callback - if specified, will call this function
    # when state updates, else it will print the maze and sleep some time
    def __init__(self, cells, printing_callback=None):
        self.printing_callback = printing_callback
        self.cells = cells
        self.current_pos = None
        self.target_pos = None

    def size(self):
        return np.shape(self.cells)[0]

    def fillWeights(self, target_x, target_y):
        self.target_pos = target_x, target_y
        for x in range(X):
            for y in range(Y):
                self.cells[x, y].x = x
                self.cells[x, y].y = y
                # use the "Manhatan length" (also: "taxicab metric", L2 norm)
                self.cells[x, y].weight = int(abs(target_x - x) + abs(target_y - y))
        # if someone chooses target x or y as a fraction it may happen that there is no
        # cell with value zero, so adjust them to be sure
        min_w = np.Inf
        for x in range(X):
            for y in range(Y):
                min_w = min(min_w, self.cells[x, y].weight)
        for x in range(X):
            for y in range(Y):
                self.cells[x, y].weight -= min_w

    def initFill(self):
        #  toSubtract = 1 if X % 2 == 0 else 0
        for x in range(X):
            for y in range(Y):
                self.cells[x, y].weight = (abs(X - 1 - 2 * x) + abs(Y - 1 - 2 * y)) // 2

    def floodFill(self, x, y):
        stack = []
        stack.append(self.cells[x][y])
        while len(stack):
            cell = stack.pop()
            neighbours = self.neighbours(cell.x, cell.y)
            minW = min([n.weight for n in neighbours])
            if not (cell.weight == minW + 1) and not (cell.weight == 0):  # cell.weight == minW + 1
                cell.weight = minW + 1
                stack.extend(neighbours)

                global MAX_STACK_SIZE
                MAX_STACK_SIZE = max(len(stack), MAX_STACK_SIZE)

                if self.printing_callback is not None:
                    print('Executing callback...')
                    self.printing_callback()
                elif FLOODFILL_DELAY > 0:
                    sleep(FLOODFILL_DELAY)
                    clear_screen()
                    printCells(self.cells, self.current_pos, self.target_pos)

    def neighbours(self, x, y):
        cell = self.cells[x][y]
        xys = []
        if not cell.N:
            xys.append((x, y + 1))
        if not cell.E:
            xys.append((x + 1, y))
        if not cell.S:
            xys.append((x, y - 1))
        if not cell.W:
            xys.append((x - 1, y))
        xys = [xy for xy in xys if 0 <= xy[0] < X and 0 <= xy[1] < Y]
        neighbours = [self.cells[x][y] for (x, y) in xys]
        return neighbours

    def __repr__(self):
        return 'Maze(%d x %d)' % (X, Y)

###

# MAZE ALGORITHM

def finished(cell):
    if cell.N and cell.E and cell.S and cell.W:
        raise Exception("Cell has walls all around! Panic!")
    return cell.weight <= 0


incr_for_direction = {
    'N': (0, 1),
    'E': (1, 0),
    'S': (0, -1),
    'W': (-1, 0),
}

opposite_dir = {
    'N': 'S',
    'E': 'W',
    'S': 'N',
    'W': 'E',
}


def incByDir(pos, direction):
    x, y = pos
    dx, dy = incr_for_direction[direction]
    return x + dx, y + dy


def xyInBounds(x, y):
    return 0 <= x < X and 0 <= y < Y


def senseWalls(cells, x, y, real_cell):
    for dir in ['N', 'E', 'S', 'W']:
        value = real_cell.wall(dir)
        cells[x, y].setWall(dir, value)
        opposite_xy = incByDir((x, y), dir)
        if xyInBounds(*opposite_xy):
            cells[opposite_xy[0], opposite_xy[1]].setWall(opposite_dir[dir], value)


def getNeighbour(cells, x, y, direction):
    if getattr(cells[x, y], direction):  # has wall
        return None
    incx, incy = incr_for_direction[direction]
    xN, yN = x + incx, y + incy
    if not xyInBounds(xN, yN):
        return None
    return cells[xN, yN]


def getNeighboursByDirection(cells, x, y):
    neighbours = {d: getNeighbour(cells, x, y, d) for d in ['N', 'E', 'S', 'W']}
    # filter out None neighbours (behind walls)
    return {d: n for d, n in neighbours.items() if n is not None}


def findBestDirectionNeighbour(cells, x, y):
    by_direction = getNeighboursByDirection(cells, x, y)
    best = min(by_direction, key=lambda d: by_direction[d].weight)
    return best, by_direction[best]


def runMaze(real_cells, robot_maze, x0, y0, xf, yf):
    """
    Move from point (x0,y0) to (xf,yf) when initially not knowing where
    the walls are. We can see only walls around current cell.
    """
    #  maze = copy.deepcopy(robot_maze)
    maze = robot_maze
    maze.fillWeights(xf, yf)
    x, y = x0, y0

    global MAX_STACK_SIZE
    MAX_STACK_SIZE = -1
    # TODO: this is brutal force, it probably can be done easier
    for xi in range(X):
        for yi in range(Y):
            #  print('Initial flood-fill for (%d, %d)...' % (x, y))
            maze.floodFill(xi, yi)

    print('init MAX_STACK_SIZE = %d' % MAX_STACK_SIZE)
    MAX_STACK_SIZE = -1

    counter = 0
    while maze.cells[x, y].weight > 0:
        senseWalls(maze.cells, x, y, real_cells[x, y])
        maze.floodFill(x, y)
        next_direction, next_cell = findBestDirectionNeighbour(maze.cells, x, y)

        if MOVE_DELAY > 0:
            clear_screen()
            printCells(maze.cells, (x, y), (xf, yf))

        #  print('Moving in direction %s (from (%d, %d, w=%d) to (%d, %d, w=%d))' %
        #        (next_direction,
        #         x, y, maze.cells[x, y].weight,
        #         next_cell.x, next_cell.y, next_cell.weight))
        assert abs(next_cell.x - x) + abs(next_cell.y - y) == 1, \
            'Moved by more than 1!'

        if MOVE_DELAY > 0:
            sleep(MOVE_DELAY)

        x, y = next_cell.x, next_cell.y
        maze.current_pos = x, y
        counter += 1

    if MOVE_DELAY > 0:
        clear_screen()
        printCells(maze.cells, (x, y), (xf, yf))

    print('MAX_STACK_SIZE = %d' % MAX_STACK_SIZE)

    return maze

###


def make_maze(w=16, h=8, print_it=False):
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["|  "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+--"] * w + ['+'] for _ in range(h + 1)]

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx == x:
                hor[max(y, yy)][x] = "+  "
            if yy == y:
                ver[y][max(x, xx)] = "   "
            walk(xx, yy)

    walk(randrange(w), randrange(h))

    if print_it:
        s = ""
        for (a, b) in zip(hor, ver):
            s += ''.join(a + ['\n'] + b + ['\n'])
        print(s)
    return ver, hor

def parse_to_my_stupid_format(ver, hor):
    n_rows, n_cols = len(hor[0]) - 1, len(hor) - 1
    cells = [Cell() for i in range(n_rows * n_cols)]

    def cell(row, col):
        return cells[n_cols * row + col]

    for row in range(n_rows):  # row = ['+--', '+--', '+--', '+--', '+--', '+--', '+--', '+--', '+']
        for col in range(n_cols):  # '+--' OR '+--', drop last (always '+')
            # set x, y values right
            cell(row, col).x = row
            cell(row, col).y = col
            # first horizontal - N and S
            if hor[row][col] == '+--':  # wall
                cell(row, col).N = True
                cell(row - 1, col).S = True
            # now vertical - E and W
            if ver[row][col] == '|  ':  # wall
                cell(row, col).W = True
                cell(row, col - 1).E = True

    cells = Cell.fromListOfCells(cells)
    #  printCells(cells)
    return cells

###

def main_run(real_cells, targets_chain):
    if MOVE_DELAY > 0:
        clear_screen()
    printCells(real_cells, current_pos=targets_chain[0], target_pos=targets_chain[1])
    print('Real maze')
    maze = Maze(Cell.emptyMaze())

    for i in range(len(targets_chain) - 1):
        posfrom, posto = targets_chain[i], targets_chain[i + 1]
        print('\nNext run from (%d, %d) to (%d, %d)' % (*posfrom, *posto))
        if INTERACTIVE:
            input('\nPRESS ANY KEY TO START NEXT RUN FROM')
        maze = runMaze(real_cells, maze, *posfrom, *posto)

def main_flood(cells):
    m = Maze(cells)
    #  m.initFill()
    m.fillWeights((X - 1) / 2, (Y - 1) / 2)
    #  m.fillWeights(2.5, 2.5) # this also works!

    # m.cells[0][2].weight = 99
    # m.cells[1][2].weight = 99
    # m.cells[0][3].weight = 99
    # m.cells[1][3].weight = 99
    # m.cells[1][2].weight = 99

    print(m)
    sys.exit()
    # print(m.neighbours(0, 2))

    m.floodFill(0, 2)
    m.floodFill(0, 3)
    m.floodFill(1, 3)
    m.floodFill(2, 3)
    m.floodFill(2, 4)
    m.floodFill(3, 4)
    m.floodFill(3, 3)
    m.floodFill(4, 3)
    m.floodFill(3, 4)
    print(m)

if __name__ == '__main__':
    # cells = [ (1, 0, 0, 1),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 1, 0, 0),
    #           (0, 0, 0, 1),       (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #           (0, 0, 0, 1),        (0, 0, 0, 0),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #           (0, 0, 0, 1),        (0, 0, 0, 0),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #           (0, 0, 1, 1),        (0, 0, 1, 0),      (0, 0, 1, 0),       (0, 0, 1, 0),       (0, 1, 1, 0),]
    #  cells = [   (1, 1, 0, 1),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 1, 0, 0),
    #              (0, 0, 0, 1),       (0, 0, 1, 0),       (0, 1, 1, 0),       (0, 0, 1, 1),       (0, 1, 0, 0),
    #              (0, 1, 0, 1),        (1, 0, 0, 1),      (1, 0, 0, 0),       (1, 0, 0, 0),       (0, 1, 0, 0),
    #              (0, 1, 0, 1),        (0, 0, 0, 1),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #              (0, 1, 1, 1),        (0, 0, 1, 1),      (0, 0, 1, 0),       (0, 0, 1, 0),       (0, 1, 1, 0),]
    #  cells = Cell.fromList(cells)

    max_size = -1
    for _ in range(1):
        cells = parse_to_my_stupid_format(*make_maze(X, Y))
        targets_chain = [
            (0, 0),
            (X // 2, Y // 2),
            (0, Y - 1),
            (X - 1, Y - 1),
            (X - 1, 0),
            (0, 0),
        ]
        main_run(cells, targets_chain)
        max_size = max(MAX_STACK_SIZE, max_size)
    print('Max registered size after all runs = %d' % max_size)
    #  cells = np.zeros([X*Y, 4], dtype=np.int)
    #  main_flood(cells)
