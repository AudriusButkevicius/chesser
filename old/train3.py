import collections
import dataclasses
import datetime
import functools
import io
import logging
import operator
import os.path
import pathlib
import random
import ctypes
import struct
from typing import Iterator, List, Iterable, Tuple, Union

import chess
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchmetrics import Accuracy
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co, Dataset
import numpy as np

import torch

torch.set_float32_matmul_precision('medium')

model_dtype = np.float32
torch_dtype = torch.float32

# class Record(ctypes.Structure):
#     """
#     white: u64,
#     pawns: u64,
#     knights: u64,
#     bishops: u64,
#     rooks: u64,
#     queens: u64,
#     kings: u64,
#     turn: u8,
#     moves: u16,
#     half_moves: u16,
#     score: f32,
#     mate: u8,
#     white_elo: u16,
#     black_elo: u16,
#     """
#
#     # (\s+)([a-z_]+): u(\d+),$
#     # $1('$2', ctypes.c_uint$3),
#
#     _pack_ = 1
#     _fields_ = (
#         ('white', ctypes.c_uint64),
#         ('pawns', ctypes.c_uint64),
#         ('knights', ctypes.c_uint64),
#         ('bishops', ctypes.c_uint64),
#         ('rooks', ctypes.c_uint64),
#         ('queens', ctypes.c_uint64),
#         ('kings', ctypes.c_uint64),
#         ('turn', ctypes.c_uint8),
#         ('moves', ctypes.c_uint16),
#         ('half_moves', ctypes.c_uint16),
#         ('score', ctypes.c_float),
#         ('mate', ctypes.c_uint8),
#         ('white_elo', ctypes.c_uint16),
#         ('black_elo', ctypes.c_uint16),
#     )
#
#     def value_to_index(self, value):
#         value_bytes = struct.pack('@Q', value)
#         byte_array = np.frombuffer(value_bytes, dtype=np.uint8)
#         bit_array = np.unpackbits(byte_array)
#         return np.nonzero(bit_array)
#
#     def board(self):
#         all_pieces = self.pawns | self.knights | self.bishops | self.rooks | self.queens | self.kings
#         black = (~self.white) & all_pieces
#         board = np.zeros(64, dtype=np.uint8)
#         for idx, piece in enumerate([self.pawns, self.knights, self.bishops, self.rooks, self.queens, self.kings]):
#             idx += 1
#             white_pieces = piece & self.white
#             board[] = idx
#
#
#         board = chess.Board()
#         board.pawns
#
#     def __repr__(self):
#         key_values = ", ".join([
#             f"{name}={getattr(self, name)}"
#             for name, _ in self._fields_
#         ])
#         return f"{self.__class__.__name__}({key_values})"

piece_types = [
    'pawns',
    'knights',
    'bishops',
    'rooks',
    'queens',
    'kings'
]

record_dtype = np.dtype(
    [
        ('white', np.uint64)
    ] + [
        (piece, np.uint64)
        for piece in piece_types
    ] + [
        ('turn', np.uint8),
        ('moves', np.uint16),
        ('half_moves', np.uint16),
        #('ep_square', np.uint8),
        #('promoted', np.uint64),
        #('castling_rights', np.uint64),
        ('score', np.float32),
        ('mate', np.uint8),
        ('white_elo', np.uint16),
        ('black_elo', np.uint16),
    ]
)
record_size = record_dtype.itemsize
#assert record_size == 72
#assert record_size == 92


def paths_to_records(paths, features_as_channels):
    for path in paths:
        with open(path, 'rb', buffering=4 * (1 << 20)) as fd:
            data = fd.read(record_size)
            while data:
                record = np.frombuffer(data, dtype=record_dtype)
                yield Converter.record_to_tensor_record(record, features_as_channels)
                data = fd.read(record_size)


class PathRecordIterator(Iterator):
    def __init__(self, paths: Iterable[pathlib.Path], features_as_channels):
        self.paths = list(paths)

        total_size = 0
        for path in paths:
            path_size = os.path.getsize(path)
            assert path_size % record_dtype.itemsize == 0
            total_size += path_size

        self.count = total_size // record_dtype.itemsize
        self.fd = None
        self.current_file_index = 0
        self.features_as_channels = features_as_channels

    def __len__(self):
        return self.count

    def __next__(self) -> T_co:
        if self.fd is None:
            try:
                next_path = self.paths[self.current_file_index]
            except IndexError:
                raise StopIteration

            self.fd = open(next_path, 'rb', buffering=4 * (1 << 20))
            self.current_file_index += 1

        data = self.fd.read(record_dtype.itemsize)
        if not data:
            self.fd.close()
            self.fd = None
            return next(self)

        record = np.frombuffer(data, dtype=record_dtype)
        return Converter.record_to_tensor_record(record, self.features_as_channels)


class Converter:
    @staticmethod
    def cp_to_win_chance(cp):
        cp = np.clip(cp, -1000, 1000)
        win_chance = 2.0 / (1.0 + np.exp(-0.00368208 * cp)) - 1.0
        return np.clip(win_chance, -1.0, 1.0)

    @staticmethod
    def win_chance_to_cp(win_chance):
        return -np.log(2 / (win_chance + 1.0) - 1) / 0.00368208

    @classmethod
    def board_to_record(cls, board: chess.Board):
        record = np.zeros(1, dtype=record_dtype)
        record['white'] = board.occupied_co[chess.WHITE]
        for piece_name, value in zip(piece_types, [
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings
        ]):
            record[piece_name] = value
        record['turn'] = 1 if board.turn == chess.WHITE else 0
        record['moves'] = board.fullmove_number
        record['half_moves'] = board.halfmove_clock
        return record

    @classmethod
    def record_to_board(cls, record: np.ndarray):
        board = chess.Board()
        board.pawns = int(record['pawns'])
        board.knights = int(record['knights'])
        board.bishops = int(record['bishops'])
        board.rooks = int(record['rooks'])
        board.queens = int(record['queens'])
        board.kings = int(record['kings'])
        all_pieces = (
            board.pawns |
            board.knights |
            board.bishops |
            board.rooks |
            board.queens |
            board.kings
        )
        board.occupied = all_pieces
        board.occupied_co[chess.WHITE] = int(record['white'])
        board.occupied_co[chess.BLACK] = int(board.occupied & ~record['white'])
        board.fullmove_number = int(record['moves'])
        board.halfmove_clock = int(record['half_moves'])
        board.turn = chess.WHITE if record['turn'] else chess.BLACK
        return board

    @classmethod
    def tensor_record_to_record(cls, boards, features, features_as_channel):
        record = np.zeros(1, dtype=record_dtype)
        board = boards.numpy()
        record = np.frombuffer(record, dtype=record_dtype)

        white = 0
        for i, piece in enumerate(piece_types):
            white_pieces = np.packbits(board[i].astype(np.uint8)).view(np.uint64)
            black_pieces = np.packbits(board[i + 6].astype(np.uint8)).view(np.uint64)
            white |= white_pieces
            record[piece] = white_pieces | black_pieces
        record['white'] = white
        if features_as_channel:
            record['turn'] = 1 if np.packbits(board[12].astype(np.uint8)).view(np.uint64) == white else 0
            record['moves'] = board[13][0][0]
            record['half_moves'] = board[13][0][1]
        else:
            features = features.numpy()
            record['turn'] = features[0]
            record['moves'] = features[1]
            record['half_moves'] = features[2]
        return record


    @classmethod
    def record_to_tensor_record(cls, record: Union[np.ndarray, bytes], features_as_channel: bool = False):
        if isinstance(record, bytes):
            record = np.frombuffer(record, dtype=record_dtype)

        one_hot_boards = np.zeros((14 if features_as_channel else 12, 64), dtype=np.uint8)

        black = 0
        for i, piece in enumerate(piece_types):
            one_hot_boards[i] = np.unpackbits((record[piece] & record['white']).view(np.uint8)).astype(np.uint8)
            one_hot_boards[i + 6] = np.unpackbits((record[piece] & ~record['white']).view(np.uint8)).astype(np.uint8)
            black |= (record[piece] & ~record['white'])

        one_hot_boards = one_hot_boards.astype(model_dtype)
        other_feature_list = []
        if features_as_channel:
            color_to_move = (record['white'] if record['turn'] else black)
            one_hot_boards[12] = np.unpackbits(color_to_move.view(np.uint8)).astype(np.uint8)
            one_hot_boards.shape = (14, 8, 8)
            one_hot_boards[13][0][0] = record['moves']
            one_hot_boards[13][0][1] = record['half_moves']
        else:
            one_hot_boards.shape = (12, 8, 8)

            other_feature_list = [
                record['turn'],
                record['moves'],
                record['half_moves']
            ]
        other_features = np.array(other_feature_list).astype(model_dtype)
        other_features.shape = len(other_feature_list)

        # https://chess.stackexchange.com/a/40254
        # https://github.com/lichess-org/lila/blob/master/modules/analyse/src/main/WinPercent.scala
        if record['mate']:
            score = cls.cp_to_win_chance(1000.0 - record['score'] if record['score'] > 0 else -1000.0 - record['score'])
        else:
            score = cls.cp_to_win_chance(np.clip(record['score'] * 100, -950, 950))

        record = {
            'board': one_hot_boards,
            'features': other_features,
            'score': score,
            # 'raw_score': record['score'],
            # 'mate': record['mate'],
        }
        return record


class EvaluationDataset(IterableDataset):
    def __init__(self, paths: Iterable[pathlib.Path], batch_size, features_as_channels):
        paths = list(sorted(paths))
        total_size = 0
        self.path_sizes = {}
        self.batch_size = batch_size
        for path in paths:
            path_size = os.path.getsize(path)
            assert path_size % record_size == 0
            self.path_sizes[path] = path_size
            total_size += path_size

        assert total_size % record_size == 0

        self.paths = paths
        self.count = total_size // record_size
        self.features_as_channels = features_as_channels
        print(f"Have {self.count} boards from {paths[0]} from {total_size} records")

    # def __len__(self):
    #     return self.count

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return PathRecordIterator(self.paths, self.features_as_channels)
        paths = list(np.array_split(self.paths, worker_info.num_workers)[worker_info.id])
        return PathRecordIterator(paths, self.features_as_channels)


class ConvBlock(nn.Module):
    def __init__(self,  channels, hidden_layer_width):
        super(ConvBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(self.channels, hidden_layer_width, 3, stride=1, padding=1, dtype=torch_dtype)
        self.bn1 = nn.BatchNorm2d(hidden_layer_width, dtype=torch_dtype)

    def forward(self, s):
        s = s.view(-1, self.channels, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=torch_dtype)
        self.bn1 = nn.BatchNorm2d(planes, dtype=torch_dtype)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=torch_dtype)
        self.bn2 = nn.BatchNorm2d(planes, dtype=torch_dtype)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, hidden_layer_width):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(hidden_layer_width, 1, kernel_size=1, dtype=torch_dtype)  # value head
        self.bn = nn.BatchNorm2d(1, dtype=torch_dtype)
        self.fc1 = nn.Linear(8 * 8, 64, dtype=torch_dtype)
        self.fc2 = nn.Linear(64, 1, dtype=torch_dtype)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 8 * 8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        return v


class EvaluationModel(pl.LightningModule):
    def __init__(
            self,
            train_data: List[pathlib.Path],
            val_data: List[pathlib.Path],
            batch_size=1024,
            learning_rate=1e-3,
            hidden_layers=10,
            hidden_layer_width=256,
            data_loader_workers=None,
            validation_file_count=3,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.data_loader_workers = data_loader_workers or os.cpu_count()
        self.use_conv = True
        self.validation_file_count = validation_file_count
        self.save_hyperparameters()

        if self.use_conv:
            layers = self.get_conv_layers()
        else:
            layers = self.get_dense_layers()

        self.seq = nn.Sequential(collections.OrderedDict(layers))

    def get_conv_layers(self) -> List[Tuple[str, any]]:
        layers: List[Tuple[str, any]] = [
            (f'entry', ConvBlock(14, self.hidden_layer_width)),
        ]
        for i in range(self.hidden_layers):
            layers.append(
                (f'hidden-{i}', ResBlock(self.hidden_layer_width, self.hidden_layer_width))
            )

        layers.append(('out', OutBlock(self.hidden_layer_width)))
        return layers

    def get_dense_layers(self) -> List[Tuple[str, any]]:
        layers: List[Tuple[str, any]] = [
            (f'entry', nn.Linear((12 * 8 * 8) + 3, self.hidden_layer_width)),
            (f'activation-entry', nn.ReLU())
        ]
        for i in range(self.hidden_layers):
            layers.append(
                (f'hidden-{i}', nn.Linear(self.hidden_layer_width, self.hidden_layer_width, dtype=torch_dtype))
            )
            layers.append(
                (f'activation-{i}', nn.ReLU())
            )

        layers.append(('out', nn.Linear(self.hidden_layer_width, 1, dtype=torch_dtype)))
        return layers

    def forward(self, board, features):
        if not self.use_conv:
            x = board
        else:
            x = torch.cat([
                torch.flatten(board, 1),
                features
            ], 1)
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        y = batch['score']
        y_hat = self.forward(batch['board'], batch['features'])
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['score']
        y_hat = self.forward(batch['board'], batch['features'])
        loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        dataset = EvaluationDataset(self.train_data, self.batch_size, self.use_conv)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_loader_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        paths = self.val_data[:]
        random.shuffle(paths)
        paths = paths[:self.validation_file_count or len(paths)]
        dataset = EvaluationDataset(paths, self.batch_size, self.use_conv)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_loader_workers, pin_memory=True, drop_last=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
            }
        }


def main():
    train_paths = list(pathlib.Path('d:/chess_split_jan').rglob('shuffled*.bin'))
    val_paths = list(pathlib.Path('d:/chess_split_feb').rglob('shuffled*.bin'))
    # val_paths = []
    import random
    random.Random(42).shuffle(val_paths)
    print(f"Have {len(train_paths)} train_paths")
    print(f"Have {len(val_paths)} val_paths")
    lr = 1e-4
    model = EvaluationModel(
        train_paths, val_paths,
        batch_size=64,
        learning_rate=lr,
        hidden_layers=8,
        hidden_layer_width=128,
        data_loader_workers=os.cpu_count(),
        validation_file_count=3,
    )
    callbacks = [
        StochasticWeightAveraging(swa_lrs=lr, device=None),
        #EarlyStopping(monitor="val_loss", verbose=True, check_on_train_epoch_end=False, patience=10),
        LearningRateMonitor(logging_interval='step', log_momentum=True),
        ModelCheckpoint(
            filename='{epoch}-{step}-{val_loss:.3f}-{train_loss_epoch:.3f}',
            save_top_k=-1,
            train_time_interval=datetime.timedelta(hours=1),
        )
    ]
    accumulate_grad_batches = 7
    tb_logger = TensorBoardLogger(save_dir="../logs_train3_2/")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=2000,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        logger=tb_logger,
        val_check_interval=2_000_000,
        limit_val_batches=100_000,
        check_val_every_n_epoch=None,
        #limit_train_batches=4,
        # limit_val_batches=4,
        log_every_n_steps=20,
    )

    trainer.fit(
        model,
        ckpt_path=sorted(pathlib.Path(r"../logs_train3_2/lightning_logs/version_6/checkpoints").rglob("*.ckpt"))[-1]
    )


if __name__ == "__main__":
    main()
