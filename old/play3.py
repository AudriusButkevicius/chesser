import pathlib
import threading
import time
import traceback
from typing import List, Union

import chess
import chess.engine
import pytorch_lightning as pl
from PyQt5 import QtGui
from PyQt5.QtGui import QFontDatabase, QColor, QColorConstants
from chess.engine import Score, PovScore, Cp, Mate
import numpy as np
import torch
import chess.pgn as pgn
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt, QSettings, QSize, QPoint

from train3 import EvaluationModel, model_dtype, Converter

# Define piece values for material count evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(QSettings.IniFormat, QSettings.SystemScope, 'Chesser', 'settings')
        self.settings.setFallbacksEnabled(False)  # File only, not registry.
        # setPath() to try to save to current working directory
        self.settings.setPath(QSettings.IniFormat, QSettings.SystemScope, './settings.ini')

        self.setFixedSize(QSize(900, 600))
        self.move(self.settings.value("pos", QPoint(100, 100)))

        self.widget_svg = QSvgWidget(parent=self)
        self.widget_svg.setGeometry(10, 10, 580, 580)
        self.list_widget = QListWidget(parent=self)
        self.list_widget.setGeometry(600, 10, 290, 580)
        self.list_widget.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.boards = []
        self.add_board(chess.Board())

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.settings.setValue("pos", self.pos())

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        board = self.boards[self.list_widget.currentRow()]
        svg = chess.svg.board(board).encode("UTF-8")
        self.widget_svg.load(svg)

    def keyPressEvent(self, evt: QtGui.QKeyEvent) -> None:
        key = evt.key()
        current = self.list_widget.currentRow()
        if (key == Qt.Key_Left or key == Qt.Key_Up) and current > 0:
            self.list_widget.setCurrentRow(current - 1)
        elif (key == Qt.Key_Right or key == Qt.Key_Down) and current < len(self.boards) - 1:
            self.list_widget.setCurrentRow(current + 1)

    def add_board(self, board, move_score1=None, move_score2=None, move_score3=None):
        board = board.copy()
        last = self.list_widget.currentRow() == len(self.boards) - 1
        self.boards.append(board)
        if board.move_stack:
            move = board.pop()
            captured = " "
            if board.is_capture(move):
                captured = board.piece_at(move.to_square).symbol().upper()
            board.push(move)
            piece = board.piece_at(move.to_square)
            piece_name = piece.symbol().upper()
            text = f"{piece_name} | {chess.SQUARE_NAMES[move.from_square]} -> {chess.SQUARE_NAMES[move.to_square]} | {captured} | {move_score1:<4} | {move_score2:<4} | {move_score3:<4}"
            item = QListWidgetItem(text)
            if piece.color == chess.BLACK:
                item.setBackground(QColorConstants.LightGray)
                # item.setForeground(QColorConstants)

            self.list_widget.addItem(item)
        else:
            item = QListWidgetItem("START")
            item.setBackground(QColorConstants.Green)
            item.setForeground(QColorConstants.Black)
            self.list_widget.addItem(item)
        if last:
            self.list_widget.setCurrentRow(len(self.boards) - 1)


class PlayThread(threading.Thread):
    def __init__(self, window: MainWindow, model_path: Union[str, pathlib.Path], depth: int, device: str):
        super().__init__(daemon=True)
        self.window = window

        self.depth = depth
        if model_path:
            self.model = EvaluationModel.load_from_checkpoint(
                model_path
            )
            self.device = torch.device(device)
            print(f"Inference device {self.device}")
            self.model.to(self.device)
            self.model.freeze()
            self.model.eval()
        else:
            self.model = None

        self.start()

    def run(self) -> None:
        engine = chess.engine.SimpleEngine.popen_uci(r"D:\stockfish-windows-2022-x86-64-avx2.exe")
        with engine:
            engine.configure({"Skill Level": 5})

            board = chess.Board()
            game = pgn.Game()
            n = 1
            try:
                while not board.is_game_over():
                    start_time = time.time()
                    # print("Starting move search")
                    best_move, best_score = self.get_best_move(board)
                    my_capture = " "
                    if board.is_capture(best_move):
                        my_capture = board.piece_at(best_move.to_square).symbol().upper()
                    # print(f"Move search took {time.time()-start_time}")
                    board.push(best_move)
                    my_move_analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
                    my_actual_score = self.boards_eval(board)[0].item()

                    my_move_score1 = self.to_score(self.convert_model_score(my_actual_score))
                    my_move_score2 = self.to_score(my_move_analysis['score'].white().wdl(model="lichess").wins - 500)
                    my_move_score3 = self.to_score(self.evaluate_board_score(board) * 10)

                    self.window.add_board(board, my_move_score1, my_move_score2, my_move_score3)

                    game = game.add_variation(best_move)

                    # print("me", n, best_move, to_score(best_score), my_move_analysis['score'].white())

                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    if result.move is None:
                        print(result)
                    if result.resigned:
                        break
                    enemy_capture = " "
                    if board.is_capture(result.move):
                        enemy_capture = board.piece_at(result.move.to_square).symbol().upper()
                    board.push(result.move)
                    game = game.add_variation(result.move)

                    stockfish_analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
                    my_score = self.boards_eval(board)[0].item()

                    opponent_move_score1 = self.to_score(self.convert_model_score(my_score))
                    opponent_move_score2 = self.to_score(
                        stockfish_analysis['score'].white().wdl(model="lichess").wins - 500)
                    opponent_move_score3 = self.to_score(self.evaluate_board_score(board) * 10)

                    duration = int(time.time() - start_time)

                    self.window.add_board(board, opponent_move_score1, opponent_move_score2, opponent_move_score3)
                    # print("stockfish", n, result.move, to_score(my_score), stockfish_analysis['score'].white())
                    ##print(result.move, result.ponder, analysis['score'].white(), my_score)
                    # print(best_score, PovScore(Cp(best_score), True), my_move_analysis['score'])
                    print(
                        '| %4s | %2s (%4s, %4s, %4s) | %s | %2s (%4s, %4s, %4s) | %s | %4ss' % (
                            n,

                            best_move.uci()[-2:],
                            my_move_score1,
                            my_move_score2,
                            my_move_score3,

                            my_capture,

                            result.move.uci()[-2:],
                            opponent_move_score1,
                            opponent_move_score2,
                            opponent_move_score3,

                            enemy_capture,

                            duration
                        )
                    )
                    # print(n, best_move.uci()[-2:], )
                    n += 1
            except Exception as e:
                traceback.print_exc()

            print(game.root())

            print(board.result())

    def boards_eval(self, *boards: chess.Board) -> torch.Tensor:
        if self.model:
            return self.evaluate_boards_model(*boards)
        result = []
        for board in boards:
            result.append(self.evaluate_board_score(board))
        return torch.from_numpy(np.array(result))

    def recursive_search(self, board, depth, maximize, alpha=-float('inf'), beta=float('inf')):
        # print(f"Entering search {depth}")
        # if maximize and board.is_game_over():
        #     return float("inf")
        if depth == 0 or board.is_game_over():
            score = self.boards_eval(board)[0].item()
            return score

        if maximize:
            best_score = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                score = self.recursive_search(board, depth - 1, False, alpha, beta)
                board.pop()
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # beta cut-off
            return best_score

        else:
            best_score = float('inf')
            for move in board.legal_moves:
                board.push(move)
                score = self.recursive_search(board, depth - 1, True, alpha, beta)
                board.pop()
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # alpha cut-off
            return best_score

    def generate_legal_moves_at_depth(self, board, depth):
        if depth == 0 or board.is_game_over():
            yield board
        else:
            for move in board.generate_legal_moves():
                board.push(move)
                for sub_board in self.generate_legal_moves_at_depth(board, depth - 1):
                    # print(f"RETURN {depth}")
                    yield sub_board.copy()
                board.pop()

    def get_best_move_batch(self, board):
        best_move = None
        best_score = -float('inf')
        for move in board.legal_moves:
            board.push(move)

            if board.can_claim_draw():
                board.pop()
                continue

            boards = list(self.generate_legal_moves_at_depth(board, self.depth))
            print(f"Got {len(boards)} boards")
            evals = self.boards_eval(*boards)
            best_move_index = torch.argmax(evals)
            score = evals[best_move_index].item()
            print(f"Got score {score}")
            if score > best_score:
                best_score = score
                best_move = move

            board.pop()

        return best_move, best_score

    def get_best_move_batch2(self, board):
        boards = []
        moves = []
        for move in board.legal_moves:
            board.push(move)

            if board.can_claim_draw():
                board.pop()
                continue

            move_boards = list(self.generate_legal_moves_at_depth(board, self.depth))
            boards.extend(move_boards)
            moves.extend([move] * len(move_boards))

            board.pop()

        print(f"Got {len(boards)} boards")
        evals = self.boards_eval(*boards)
        best_move_index = torch.argmax(evals)
        best_score = evals[best_move_index].item()
        best_move = moves[best_move_index]
        print(f"Got score {best_score} for {best_move}")

        return best_move, best_score

    def get_best_move(self, board):
        best_move = None
        best_score = -float('inf')
        for move in board.legal_moves:
            board.push(move)

            if board.can_claim_draw():
                board.pop()
                continue

            #score = self.boards_eval(board)[0].item()

            score = self.recursive_search(board, self.depth, False)
            if score > best_score:
                best_score = score
                best_move = move

            board.pop()

        return best_move, best_score

    @staticmethod
    def convert_model_score(value):
        value = Converter.win_chance_to_cp(value)
        #value = int(value * 100)
        value = PovScore(Cp(value), chess.WHITE)
        return value.white().wdl(model="lichess").wins - 500

    @staticmethod
    def to_score(value):
        if value >= 0:
            return f"+{int(value)}"
        return str(int(value))

    def evaluate_boards_model(self, *boards: chess.Board):
        if not boards:
            return []

        batch_size = len(boards)
        one_hot_boards = np.zeros((batch_size, 14 if self.model.use_conv else 12, 8, 8), dtype=model_dtype)
        other_features = np.zeros((batch_size, 3 if not self.model.use_conv else 0), dtype=model_dtype)
        for n, board in enumerate(boards):
            np_record = Converter.board_to_record(board)
            record = Converter.record_to_tensor_record(np_record, self.model.use_conv)
            one_hot_boards[n] = record['board']
            other_features[n] = record['features']

        # one_hot_boards = one_hot_boards.astype(np.float32)
        # one_hot_boards.shape = (batch_size, 14 if self.model.use_conv else 12, 8, 8)
        # other_features.shape = (batch_size, 3 if not self.model.use_conv else 0)

        one_hot_boards_tensor = torch.from_numpy(one_hot_boards).to(self.device)
        other_features_tensor = torch.from_numpy(other_features).to(self.device)

        with torch.no_grad():
            return self.model.forward(one_hot_boards_tensor, other_features_tensor)

    @staticmethod
    def evaluate_board_score(board: chess.Board):
        # Evaluate material count
        material_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                material_count += PIECE_VALUES[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)

        # Evaluate piece mobility
        mobility_count = 0
        for move in board.legal_moves:
            mobility_count += 1 if board.piece_at(move.from_square).piece_type != chess.PAWN else 0.5

        # Evaluate pawn structure
        pawn_structure_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    pawn_structure_count += 1 if square in [chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2,
                                                            chess.G2, chess.H2] else 0
                else:
                    pawn_structure_count -= 1 if square in [chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7,
                                                            chess.G7, chess.H7] else 0

        # Evaluate king safety
        king_square = board.king(chess.WHITE)
        king_safety_count = 0
        if king_square is not None:
            # Check if king is castled
            if king_square == chess.G1 or king_square == chess.G8:
                king_safety_count += 1
            # Check if king has pawn shield
            pawn_shield = [
                square
                for square in chess.SQUARES
                if square // 8 == king_square // 8 - 1 and abs(square % 8 - king_square % 8) <= 1
            ]
            if all(
                    board.piece_at(square) is None or board.piece_at(square).piece_type == chess.PAWN
                    for square in pawn_shield
            ):
                king_safety_count += 1

        # Evaluate space control
        space_count = 0
        for square in [chess.D4, chess.E4, chess.D5, chess.E5]:
            piece_at_square = board.piece_at(square)
            if piece_at_square is None:
                continue
            if piece_at_square.color == chess.WHITE:
                space_count += 1
            else:
                space_count -= 1

        # Evaluate piece coordination
        coordination_count = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces = board.pieces(piece_type, chess.WHITE)
            black_pieces = board.pieces(piece_type, chess.BLACK)
            white_count = sum(1 for square in white_pieces if len(board.attacks(square)) > 0)
            black_count = sum(1 for square in black_pieces if len(board.attacks(square)) > 0)
            coordination_count += white_count - black_count

        # Combine all factors with weights to get final evaluation score
        evaluation = material_count
        evaluation += 0.1 * mobility_count
        evaluation += 0.5 * pawn_structure_count
        evaluation += 0.2 * king_safety_count
        evaluation += 0.2 * space_count
        evaluation += 0.1 * coordination_count

        return evaluation


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    use_gpu = True
    play_thread = PlayThread(
        window,
        model_path=list(
            pathlib.Path(r"../logs_train3_2/lightning_logs/version_5/checkpoints").rglob("*.ckpt")
        )[0],
        depth=0,
        device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    )
    app.exec()
