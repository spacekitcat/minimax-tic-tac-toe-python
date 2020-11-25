import sys
import qprompt
import random
from enum import Enum
from copy import deepcopy
from itertools import groupby
import numpy as np
from scipy.ndimage import rotate
from rich.progress import (
    BarColumn,
    TimeRemainingColumn,
    Progress,
)

class Player(Enum):
  X = 1
  O = 2

def initialize_board():
  return [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]

def standardPositionRenderer(position, space):
  position_str = '\033[96m | \033[0m'
  if position == Player.X.value:
    position_str += f'\033[91m{Player.X.name}\033[0m'
  elif position == Player.O.value:
    position_str += f'\033[92m{Player.O.name}\033[0m'
  else:
    position_str += f'{space}'

  return position_str

def row_separator_renderer(column_count):
  separator = '\033[96m -\033[0m'
  for j in range(0, column_count):
    separator += '\033[96m----\033[0m'
  separator += '\n'

  return separator

def row_terminator_renderer():
  return '\033[96m | \033[0m\n'

def value_is_neutral(value):
  return value != Player.X.value and value != Player.O.value

def display_board(board, indent=0):
  space = 0
  for i in range(0, indent):
    sys.stdout.write('\t')
  for b in range(0, indent):
    sys.stdout.write('\t')
  sys.stdout.write(row_separator_renderer(len(board[0])))
  for i in board:
    for b in range(0, indent):
      sys.stdout.write('\t')
    for j in i:
      sys.stdout.write(standardPositionRenderer(j, space))
      if value_is_neutral(j):
        space = space + 1
    sys.stdout.write(row_terminator_renderer())
    for b in range(0, indent):
      sys.stdout.write('\t')

    sys.stdout.write(row_separator_renderer(len(i)))

def get_available_moves(board):
  available_moves = []
  for i in range(0, len(board)):
    for j in range(0, len(board)):
      if board[i][j] == -1:
        available_moves.append((i, j))

  return available_moves

def create_menu(move_list):
  menu = qprompt.Menu()
  for i in range(0, len(move_list)):
    menu.add(str(i), move_list[i])

  return menu

def get_player_to_move(move_counter):
  return move_counter % 2 + 1

def menu_prompt(move_list, move_counter):
  menu = create_menu(move_list)
  return menu.show(returns="desc", header=str.format("Turn {}, {} to move", move_counter, Player(get_player_to_move(move_counter))))

def set_board_position(board, position, value):
  x, y = position
  board[x][y] = value

def detect_horizontal_win_states(board):
  for row in board:
    if value_is_neutral(row[0]):
      continue

    # group row by unique values, if all the values are the same, the iterator will return one value followed by False
    grouped_iterator = groupby(row)
    if next(grouped_iterator, True) and not next(grouped_iterator, False):
      return row[0]

  return None

def transpose_board(board):
  return zip(*board)

def detect_win_state(board):
  orthogonal_win_state = detect_horizontal_win_states(transpose_board(board)) or detect_horizontal_win_states(board)
  diagonal_win_state = detect_horizontal_win_states([np.diag(board)]) or detect_horizontal_win_states([np.diag(np.flip(board, axis=1))])

  return orthogonal_win_state or diagonal_win_state

def calculate_board_fitness(board, player):
  opponent = Player.X
  if player == Player.X:
    opponent = Player.O

  if detect_win_state(board) == None:
    return 0.25
  elif Player(detect_win_state(board)) == player:
    return 1.0
  elif Player(detect_win_state(board)) == opponent:
    return - 1.0
  elif get_current_move(board) == 9:
    return 0.5

def get_current_move(board):
  move = 0
  for row in board:
    for cell in row:
      if cell in Player._value2member_map_:
        move += 1

  return move

def get_current_player(board):
    if get_current_move(board) % 2 == 1:
      return Player.O

    return Player.X

class Node():
  def __init__(self, board):
    self.board = board
    self.move = None
    self.player = Player.O

  def get_player(self):
    return get_current_player(self.board)

  def copy(self, move):
    copy = deepcopy(self)
    set_board_position(copy.get_board(), move, self.get_player().value)
    copy.move = move
    return copy

  def get_board(self):
    return self.board

  def has_win_state(self):
    return detect_win_state(self.get_board()) != None

  def get_children(self):
    children = []
    for move in get_available_moves(self.get_board()):
      children.append(self.copy(move))

    return children

  def get_heuristic(self):
    return calculate_board_fitness(self.board, self.player)

  def get_move(self):
    return self.move

  def get_child_count(self):
    return len(self.get_children())

  def has_children(self):
    return self.get_child_count() > 0

def minimax(node, depth, maximizingPlayer):
  children = node.get_children()
  if node.has_win_state() or depth == 0 or len(children) == 0:
    return node.get_heuristic()
  
  if maximizingPlayer:
    value = -1000
    for child in children:
      value = max(value, minimax(child, depth - 1, False))
    return value
  else:
    value = 1000
    for child in children:
      value = min(value, minimax(child, depth - 1, True))
    return value

def computer_compute_move(_board):
  progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn()
  )

  root = Node(_board)
  task1 = progress.add_task("[red] BEEP BOP", total=root.get_child_count())

  if root.has_children():
    best_candidate_child = root.get_children()[0]
    best_candidate_child_score = -1

    with progress:
      for child in root.get_children():
        progress.update(task1, advance=1, refresh=True)
        current_child_score = minimax(child, 200, False)

        if current_child_score > best_candidate_child_score:
          best_candidate_child = child
          best_candidate_child_score = current_child_score

      return best_candidate_child.get_move()

def computer_random_move(board, move_list, player):
  return random.choice(move_list)

_game_counter = 0
_draws = 0
_x_wins = 0
_o_wins = 0

_games_to_play = 10

while (_game_counter < _games_to_play):
  _board = initialize_board()
  display_board(_board)

  _move_list = get_available_moves(_board)
  while (_move_list and len(_move_list) > 0):
    if Player(get_player_to_move(get_current_move(_board))) == Player.X:
      # move = menu_prompt(_move_list, get_current_move(_board))
      move = computer_random_move(_board, _move_list, Player(get_player_to_move(get_current_move(_board))))
    else:
      move = computer_compute_move(_board)

    set_board_position(_board, move, get_player_to_move(get_current_move(_board)))
    display_board(_board)
    
    if detect_win_state(_board) != None:
      print()
      print(f'\tWIN STATE DETECTED FOR {Player(get_player_to_move(get_current_move(_board) - 1))}')
      if Player(get_player_to_move(get_current_move(_board) - 1)) == Player.X:
        _x_wins += 1

        # Force early exit if the AI manages to lose. It shouldn't do worse than a draw.
        # This ensures the game history is captured in the terminal.
        _game_counter = _games_to_play
      else:
        _o_wins += 1
      break

    _move_list = get_available_moves(_board)
  
  if not detect_win_state(_board):
    print()
    _draws += 1
    print(f'\tDRAW STATE DETECTED')

  print()
  _gc_disp = _game_counter + 1
  print(f'\t  GAME {_gc_disp:08}')
  print(f'\t DRAWS {_draws:08}')
  print(f'\tX WINS {_x_wins:08}')
  print(f'\tO WINS {_o_wins:08}')
  print()
  
  _game_counter += 1

