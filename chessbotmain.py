import random
import time

import chess
import numpy as np
from numpy import array
from tkinter import *
from tkinter import ttk
import threading

class ChessBot(object):
    def __init__(self, side, depth, prune_value, check_weight, piece_loss_weight, piece_take_weight):
        self.board = chess.Board()
        self.side = side
        self.depth = depth
        self.prune_value = prune_value
        self.checkmate_value = 48
        self.final_value = 0
        self.moves = list()
        self.check_w = check_weight
        self.piece_loss_w = piece_loss_weight
        self.piece_take_w = piece_take_weight
        self.piece_value = {chess.PAWN: 1,
                            chess.BISHOP: 4,
                            chess.KNIGHT: 4,
                            chess.ROOK: 6,
                            chess.QUEEN: 11,
                            chess.KING: 1}

    def get_board_value(self, board, value_dict):
        piece_map = board.piece_map()
        white_count = 0
        black_count = 0
        for i in piece_map:
            piece = piece_map[i]
            if piece.color == chess.WHITE:
                white_count += value_dict[piece.piece_type]
            if piece.color == chess.BLACK:
                black_count += value_dict[piece.piece_type]
        if board.is_checkmate():
            if board.outcome().winner == chess.BLACK:
                black_count += self.checkmate_value
                white_count -= self.checkmate_value * self.check_w
            else:
                white_count += self.checkmate_value
                black_count -= self.checkmate_value * self.check_w
        white_count *= self.piece_loss_w
        black_count *= self.piece_loss_w
        white_count -= black_count * self.piece_take_w
        black_count -= white_count * self.piece_take_w
        return white_count, black_count

    def run_recursion(self, paths, board):
        if board.is_checkmate() or paths == self.depth:
            w_c, b_c = self.get_board_value(board, self.piece_value)
            if self.side == chess.WHITE:
                self.global_final_count += w_c
            else:
                self.global_final_count += b_c
            return
        else:
            paths += 1
            board_value = list()
            boards = list()
            for move in list(board.legal_moves):
                board.push(move)
                w_c, b_c = self.get_board_value(board, self.piece_value)
                if self.side == chess.WHITE:
                    board_value.append(w_c)
                else:
                    board_value.append(b_c)
                boards.append(board)
                board.pop()
            for i in range(len(list(board.legal_moves)) // self.prune_value):
                min_index = board_value.index(min(board_value))
                boards.pop(min_index)
                board_value.pop(min_index)
            for i in boards:
                self.run_recursion(paths, i)
        return

    def best_move(self):
        self.moves = list()
        legal_moves = list(self.board.legal_moves)
        for i in range(len(legal_moves)):
            print(f"{i} out of {len(legal_moves)} base paths starting.")
            self.board.push(legal_moves[i])
            self.global_final_count = 0
            self.run_recursion(0, self.board)
            self.moves.append(self.global_final_count)
            self.board.pop()
            print(f"{i} out of {len(legal_moves)} base paths complete.")
        if len(set(self.moves)) == 1:
            return self.random_move(self.board)
        else:
            return legal_moves[numpy.argmax(array(self.moves))]

    def random_move(self, board):
        return random.choice(list(board.legal_moves))

    def make_matrix(self, board):
        pgn = board.epd()
        matix = []
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        for row in rows:
            array_segs = []
            for obj in row:
                if obj.isdigit():
                    for i in range(0, int(obj)):
                        array_segs.append('.')
                else:
                    array_segs.append(obj)
            matix.append(array_segs)
        return matix

    def print_board(self, board):
        print("  a b c d e f g h")
        matrix_board = self.make_matrix(board)
        for i in range(8):
            matrix_board[i].insert(0, str(8 - i))
        for i in range(8):
            line = ""
            for j in range(9):
                line += (matrix_board[i][j] + " ")
            print(line)

    def run(self):
        while not self.board.is_game_over():
            print()
            if self.side == chess.WHITE:
                self.board.push(self.best_move())
            self.print_board(self.board)
            print()
            while True:
                player_move = input()
                try:
                    self.board.push_san(player_move)
                    break
                except:
                    pass
            self.print_board(self.board)
            if self.side == chess.BLACK:
                self.board.push(self.best_move())


class ChessAI(object):
    def __init__(self):
        self.NINPUTS = 2
        self.NOUTPUTS = 1
        self.NHIDDEN = 1
        self.HIDDENSIZE = 15

        self.inputLayer = np.zeros((self.NINPUTS, self.HIDDENSIZE))
        self.interLayers = np.zeros((self.HIDDENSIZE, self.HIDDENSIZE, self.NHIDDEN))
        self.outputLayer = np.zeros((self.HIDDENSIZE, self.NOUTPUTS))

        self.inputBias = np.zeros((self.HIDDENSIZE))
        self.interBiases = np.zeros((self.HIDDENSIZE, self.NHIDDEN))
        self.outputBias = np.zeros((self.NOUTPUTS))

        self.inputValues = np.zeros((self.NINPUTS))
        self.outputValues = np.zeros((self.NOUTPUTS))


class ChessGame(object):
    def __init__(self):
        self.chess_bot = ChessBot(chess.BLACK, 3, 5, 1.8, 1.5, 1.3)
        self.window = Tk()
        self.window.geometry("1000x1000")
        self.frm = ttk.Frame(self.window, padding=10)
        self.frm.grid()
        self.buttons = list()
        self.saved_str_move = ""
        self.threads = list()

    def create_board(self):
        self.buttons = list()
        x_key = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        board = self.chess_bot.board
        self.chess_bot.print_board(board)
        board_matrix = self.chess_bot.make_matrix(board)
        for x in range(len(board_matrix)):
            for y in range(len(board_matrix[x])):
                button = ttk.Button(self.frm, text = board_matrix[y][x], command= lambda str = f"{x_key[x]}{8 - y}": self.add(str))
                button.grid(column=x, row=y, ipadx=10, ipady=30)
                self.buttons.append(button)


    def add(self, str):
        print("Button Clicked")
        if len(self.saved_str_move) == 2 or self.saved_str_move == "":
            self.saved_str_move += str
            print(self.saved_str_move)
        else:
            pass

    def play(self):
        while not self.chess_bot.board.is_game_over():
            self.create_board()
            if self.chess_bot.side == self.chess_bot.board.turn:
                self.chess_bot.board.push(self.chess_bot.best_move())
            else:
                while True:
                    if len(self.saved_str_move) == 4:
                        try:
                            self.chess_bot.board.push_san(self.saved_str_move)
                            self.saved_str_move = ""
                            break
                        except:
                            pass

    def run(self):
        print("Start")
        self.threads.append(threading.Thread(target=self.play, name="PLAY"))
        for tt in self.threads:
            tt.start()
            print("Threads start")
        print("Starting mainloop")
        self.window.mainloop()

chessg = ChessGame()

chessg.run()