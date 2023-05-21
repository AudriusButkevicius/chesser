import pathlib
import time

import chess
import chess.engine
from chess.engine import Score, PovScore, Cp, Mate
import numpy as np
import torch
import chess.pgn as pgn

from train2 import EvaluationModel

engine = chess.engine.SimpleEngine.popen_uci(r"D:\stockfish-windows-2022-x86-64-avx2.exe")
with engine:
    engine.configure({"Skill Level": 1})
    model = EvaluationModel.load_from_checkpoint(list(pathlib.Path(
        r"../lightning_logs/version_1/checkpoints").rglob("*.ckpt"))[0])
    model.freeze()
    model.eval()

    def board_eval(board: chess.Board):
        #board_array = np.zeros(64, dtype=np.float32)

        piece_types = [
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings
        ]

        one_hot_boards = np.zeros((12, 64), dtype=np.uint8)

        white = np.array([board.occupied_co[chess.WHITE]], dtype=np.uint64)
        for i, piece in enumerate(piece_types):
            piece = np.array([piece], dtype=np.uint64)
            one_hot_boards[i] = np.unpackbits((piece & white).view(np.uint8)).astype(np.uint8)
            one_hot_boards[i + 6] = np.unpackbits((piece & ~white).view(np.uint8)).astype(np.uint8)

        one_hot_boards = one_hot_boards.astype(np.float32)
        one_hot_boards.shape = (1, 12, 8, 8)

        other_feature_list = [
            1 if board.turn == chess.WHITE else 0,
            board.fullmove_number,
            board.halfmove_clock,
        ]

        other_features = np.array([other_feature_list]).astype(np.float32)
        other_features.shape = (1, len(other_feature_list))

        one_hot_boards_tensor = torch.from_numpy(one_hot_boards)
        other_features_tensor = torch.from_numpy(other_features)
        return model(one_hot_boards_tensor, other_features_tensor)[0].item()


    def recursive_search(board, depth, maximize, alpha=-float('inf'), beta=float('inf')):
        #print(f"Entering search {depth}")
        if maximize and board.is_game_over():
            return float("inf")
        if depth == 0 or board.is_game_over():
            return board_eval(board)

        if maximize:
            best_score = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                score = recursive_search(board, depth - 1, False, alpha, beta)
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
                score = recursive_search(board, depth - 1, True, alpha, beta)
                board.pop()
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # alpha cut-off
            return best_score

    def get_best_move(board):
        best_move = None
        best_score = -float('inf')
        for move in board.generate_legal_moves():
            board.push(move)
            score = recursive_search(board, 2, False)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def model_score(value):
        value = PovScore(Cp(value), True)
        return value.white().wdl(model="lichess").wins - 500

    def to_score(value):
        if value >= 0:
            return f"+{value}"
        return str(value)


    board = chess.Board()
    game = pgn.Game()
    n = 1
    try:
        while not board.is_game_over():
            start_time = time.time()
            #print("Starting move search")
            best_move, best_score = get_best_move(board)
            my_capture = " "
            if board.is_capture(best_move):
                my_capture = board.piece_at(best_move.to_square).symbol()
            #print(f"Move search took {time.time()-start_time}")
            board.push(best_move)
            game = game.add_variation(best_move)

            my_move_analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
            # print("me", n, best_move, to_score(best_score), my_move_analysis['score'].white())

            result = engine.play(board, chess.engine.Limit(time=0.01))
            if result.move is None:
                print(result)
            if result.resigned:
                break
            enemy_capture = " "
            if board.is_capture(result.move):
                enemy_capture = board.piece_at(result.move.to_square).symbol()
            board.push(result.move)
            game = game.add_variation(result.move)

            stockfish_analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
            my_score = board_eval(board)
            # print("stockfish", n, result.move, to_score(my_score), stockfish_analysis['score'].white())
            ##print(result.move, result.ponder, analysis['score'].white(), my_score)
            #print(best_score, PovScore(Cp(best_score), True), my_move_analysis['score'])
            print(
                '| %4s | %2s (%4s, %4s) | %s | %2s (%4s, %4s) | %s | %4ss' % (
                    n,

                    best_move.uci()[-2:],
                    to_score(model_score(best_score)),
                    to_score(my_move_analysis['score'].white().wdl(model="lichess").wins - 500),

                    my_capture,

                    result.move.uci()[-2:],
                    to_score(model_score(my_score)),
                    to_score(stockfish_analysis['score'].white().wdl(model="lichess").wins - 500),

                    enemy_capture,

                    int(time.time() - start_time)
                )
            )
            # print(n, best_move.uci()[-2:], )
            n += 1
    except Exception as e:
        print(e)

    print(game.root())

    print(board.result())
