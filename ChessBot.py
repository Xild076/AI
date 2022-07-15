import chess
import numpy
from numpy import array
import random


class ChessBot(object):
    def __init__(self, side, depth, prune_value):
        self.board = chess.Board()
        self.side = side
        self.depth = depth
        self.prune_value = prune_value
        self.checkmate_value = 48
        self.final_value = 0
        self.moves = list()
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
                white_count -= self.checkmate_value
            else:
                white_count += self.checkmate_value
                black_count -= self.checkmate_value
        return white_count, black_count

    def run_recursion(self, paths, board):
        if board.is_checkmate():
            return
        if board.turn == self.side:
            paths += 1
        board_value = list()
        boards = list()
        for move in list(board.legal_moves):
            board.push(move)
            self.get_board_value(board, self.piece_value)
            board_value.append(self.get_board_value(self.piece_value))
            boards.append(board)
            board.pop()
        for i in range(len(list(board.legal_moves)//self.prune_value)):
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
            self.run_recursion(0)
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

chess_bot = ChessBot(chess.WHITE, 3, 3)