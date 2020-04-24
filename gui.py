#!/usr/bin/python3

import copy
import sys

from PyQt5.QtCore import QEventLoop, QObject, QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QTransform
from PyQt5.QtWidgets import QApplication, QWidget

import maze


# monkey patch the module to work with graphical view
def thread_sleep(sec):
    QThread.msleep(int(sec * 1000))


def no_clear_screen():
    pass


def no_input(*args):
    pass


maze.sleep = thread_sleep
maze.clear_screen = no_clear_screen
#  maze.input = no_input


def monkey_patch_printCells(runner):
    def new_printCells(cells, current_pos=None, target_pos=None):
        runner.requestPrintCells(cells, current_pos, target_pos)
    maze.printCells = new_printCells


###

class MazeRunner(QObject):
    printSignal = pyqtSignal('PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject')

    def main(self):
        while True:
            real_cells = maze.parse_to_my_stupid_format(*maze.make_maze(maze.X, maze.Y))
            targets_chain = [
                (0, 0),
                (maze.X // 2, maze.Y // 2),
                (0, maze.Y - 1),
                (maze.X - 1, maze.Y - 1),
                (maze.X - 1, 0),
                (0, 0),
            ]
            maze.main_run(real_cells, targets_chain)
            input('\nPRESS ANY KEY TO START NEXT RUN FROM')

    def requestPrintCells(self, cells, current_pos=None, target_pos=None):
        self.printSignal.emit(cells, current_pos, target_pos)


class MazeWidget(QWidget):
    def __init__(self, size_px=600, wall_width=.1, font_size=.2):
        assert maze.X == maze.Y
        self.maze_size = maze.X
        self.size_px = size_px
        self.cell_width = size_px / self.maze_size
        self.wall_width = wall_width * self.cell_width
        self.font_size = font_size * self.cell_width
        super().__init__()
        self.initUI()
        self.setFixedSize(self.size())
        #  # for display
        #  self.next_mazes = []
        #  self.action_timer = QTimer()
        #  self.action_timer.timeout.connect(self.nextAction)
        #  self.action_timer.setSingleShot(True)
        #  self.action_timer.start(300)
        #  # last values of weights
        #  self.last_cells = copy.deepcopy(self.maze.cells)
        self.cells = None
        self.current_pos = None
        self.target_pos = None

        self.runner = MazeRunner()
        self.runnerThread = QThread()
        self.runner.moveToThread(self.runnerThread)
        self.runner.printSignal.connect(self.updateMaze)
        self.runnerThread.start()
        monkey_patch_printCells(self.runner)

        self.start_timer = QTimer()
        self.start_timer.timeout.connect(self.runner.main)
        self.start_timer.setSingleShot(True)
        self.start_timer.start(500)

    def updateMaze(self, cells, current_pos=None, target_pos=None):
        self.cells = cells
        self.current_pos = current_pos
        self.target_pos = target_pos
        self.update()

    #  def nextAction(self):
    #      #  for action in self.actions:
    #      #      action()
    #      if not len(self.next_mazes):
    #          return
    #      print('Action')
    #      self.maze = self.next_mazes.pop(0)
    #      self.update()
    #      self.action_timer.start()

    def initUI(self):
        # (width, height)
        self.setGeometry(700, 100, self.size_px, self.size_px)
        self.setWindowTitle('Maze')
        self.show()

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        self.drawMaze(painter)
        painter.end()

    def drawMaze(self, painter):
        if self.cells is not None:
            for x in range(self.maze_size):
                for y in range(self.maze_size):
                    is_current = False
                    is_target = False
                    if self.target_pos[0] == x and self.target_pos[1] == y:
                        is_target = True
                    if self.current_pos[0] == x and self.current_pos[1] == y:
                        is_current = True
                    self.drawCell(painter, self.cells[x][y],
                                  is_current=is_current, is_target=is_target)

    def drawCell(self, painter, cell, is_current=False, is_target=False):
        painter.setTransform(QTransform().translate(
            cell.x * self.cell_width,
            # account for different representations:
            # qt uses x0, y0 top-left, I use x0, y0 bottom left (like a plot)
            (self.maze_size - cell.y - 1) * self.cell_width))
        cw, ww = self.cell_width, self.wall_width

        # background
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        painter.drawRect(0, 0, int(cw), int(cw))

        # weight changing as gray
        if hasattr(cell, 'last_weight') and cell.last_weight != cell.weight:
            painter.setPen(QPen(Qt.gray, 0, Qt.SolidLine))
            painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
            #  painter.drawEllipse(cw/4, cw/4, cw/2, cw/2)
            painter.drawRect(0, 0, int(cw), int(cw))
        cell.last_weight = cell.weight
        # red-filled circle means target
        if is_target:
            painter.setPen(QPen(Qt.red, 0, Qt.SolidLine))
            painter.setBrush(QBrush(QColor(250, 200, 200), Qt.SolidPattern))
            painter.drawEllipse(int(cw / 4), int(cw / 4), int(cw / 2), int(cw / 2))
        # not filled circle means mouse
        if is_current:
            painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(cw / 4), int(cw / 4), int(cw / 2), int(cw / 2))

        def draw_wall(x0, y0, w, h):  # in local px
            painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
            painter.drawRect(int(x0), int(y0), int(w), int(h))
            #  print('drawing (%d, %d): x0=%d, y0=%d, w=%d, h=%d' % (cell.x, cell.y, x0, y0, w, h))

        if cell.N:
            draw_wall(0, 0, cw, ww)
        if cell.S:
            draw_wall(0, cw - ww, cw, ww)
        if cell.E:
            draw_wall(cw - ww, 0, ww, cw)
        if cell.W:
            draw_wall(0, 0, ww, cw)

        # gray lines
        painter.setPen(QPen(Qt.gray, 2, Qt.SolidLine))
        painter.drawLine(0, 0, int(cw), 0)
        painter.drawLine(0, 0, 0, int(cw))
        painter.drawLine(int(cw), int(cw), 0, int(cw))
        painter.drawLine(int(cw), int(cw), int(cw), 0)

        # text - cell weight
        painter.setPen(Qt.black)
        painter.setFont(QFont('Decorative', int(self.font_size)))
        painter.drawText(QRect(0, 0, int(cw), int(cw)), Qt.AlignCenter, str(cell.weight))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    #  # cells = [ (1, 0, 0, 1),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 1, 0, 0),
    #  #           (0, 0, 0, 1),       (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #  #           (0, 0, 0, 1),        (0, 0, 0, 0),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #  #           (0, 0, 0, 1),        (0, 0, 0, 0),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #  #           (0, 0, 1, 1),        (0, 0, 1, 0),      (0, 0, 1, 0),       (0, 0, 1, 0),       (0, 1, 1, 0),]
    #  cells = [   (1, 1, 0, 1),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 0, 0, 0),       (1, 1, 0, 0),
    #              (0, 0, 0, 1),       (0, 0, 1, 0),       (0, 1, 1, 0),       (0, 0, 1, 1),       (0, 1, 0, 0),
    #              (0, 1, 0, 1),        (1, 0, 0, 1),      (1, 0, 0, 0),       (1, 0, 0, 0),       (0, 1, 0, 0),
    #              (0, 1, 0, 1),        (0, 0, 0, 1),      (0, 0, 0, 0),       (0, 0, 0, 0),       (0, 1, 0, 0),
    #              (0, 1, 1, 1),        (0, 0, 1, 1),      (0, 0, 1, 0),       (0, 0, 1, 0),       (0, 1, 1, 0),]
    #
    #  m = Maze(cells)
    #  m.initFill()
    #  tmp_m = copy.deepcopy(m)
    #
    #  maze_states = []
    #  def add_new_state():
    #      maze_states.append(copy.deepcopy(tmp_m))
    #
    #  tmp_m.printing_callback = add_new_state
    #  m.printing_callback = lambda: 2 + 2  # supress printing and sleep
    #
    #  # m.cells[0][2].weight = 99
    #  # m.cells[1][2].weight = 99
    #  # m.cells[0][3].weight = 99
    #  # m.cells[1][3].weight = 99
    #  # m.cells[1][2].weight = 99
    #
    #  # print(m.neighbours(0, 2))
    #
    #  def fill(x, y):
    #      tmp_m.floodFill(x, y)
    #
    #  #  fills = [
    #  fill(0, 2)#,
    #  fill(0, 3)#,
    #  fill(1, 3)#,
    #  fill(2, 3)#,
    #  fill(2, 4)#,
    #  fill(3, 4)#,
    #  fill(3, 3)#,
    #  fill(4, 3)#,
    #  fill(3, 4)#,
    #  #  ]
    #
    #  maze = MazeWidget(m)
    #  maze.next_mazes.extend(maze_states)

    m = MazeWidget()
    sys.exit(app.exec_())
