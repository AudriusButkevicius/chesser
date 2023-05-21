import collections
import dataclasses
import functools
import io
import logging
import operator
import os.path
import pathlib
import random
import ctypes
import struct
from typing import Iterator, List, Iterable, Tuple

import chess
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner
from torchmetrics import Accuracy
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
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

record_dtype = np.dtype([
    ('white', np.uint64)
    ] + [
        (piece, np.uint64)
        for piece in piece_types
    ] + [
    ('turn', np.uint8),
    ('moves', np.uint16),
    ('half_moves', np.uint16),
    ('score', np.float32),
    ('mate', np.uint8),
    ('white_elo', np.uint16),
    ('black_elo', np.uint16),
])


@dataclasses.dataclass
class Record:
    one_hot_board: np.array
    turn: np.array
    moves: np.array
    half_moves: np.array
    score: np.array
    mate: np.array


class PathRecordIterator(Iterator):
    def __init__(self, paths: Iterable[pathlib.Path]):
        self.paths = list(paths)

        total_size = 0
        for path in paths:
            path_size = os.path.getsize(path)
            assert path_size % record_dtype.itemsize == 0
            total_size += path_size

        self.count = total_size // record_dtype.itemsize
        self.fd = None
        self.current_file_index = 0

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

        return self.to_record(data)

    @classmethod
    def to_record(cls, data: bytes):
        assert len(data) == record_dtype.itemsize

        record = np.frombuffer(data, dtype=record_dtype)

        # all_pieces = functools.reduce(operator.or_, [
        #     record[piece]
        #     for piece in piece_types
        # ], 0)
        # black = (~record['white']) & all_pieces

        one_hot_boards = np.zeros((12, 64), dtype=np.uint8)

        for i, piece in enumerate(piece_types):
            one_hot_boards[i] = np.unpackbits((record[piece] & record['white']).view(np.uint8)).astype(np.uint8)
            one_hot_boards[i+6] = np.unpackbits((record[piece] & ~record['white']).view(np.uint8)).astype(np.uint8)

        one_hot_boards = one_hot_boards.astype(model_dtype)
        one_hot_boards.shape = (12, 8, 8)

        other_feature_list = [
            record['turn'],
            record['moves'],
            record['half_moves']
        ]
        other_features = np.array(other_feature_list).astype(model_dtype)
        other_features.shape = len(other_feature_list)

        return {
            'board': one_hot_boards,
            'features': other_features,
            'score': record['score'].astype(model_dtype)
        }


class EvaluationDataset(IterableDataset):
    def __init__(self, paths: Iterable[pathlib.Path]):
        paths = list(sorted(paths))
        total_size = 0
        self.path_sizes = {}
        for path in paths:
            path_size = os.path.getsize(path)
            assert path_size % record_dtype.itemsize == 0
            self.path_sizes[path] = path_size
            total_size += path_size

        assert total_size % record_dtype.itemsize == 0

        self.paths = paths
        self.count = total_size // record_dtype.itemsize
        print(f"Have {self.count} boards")

    def __len__(self):
        return self.count

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return PathRecordIterator(self.paths)
        paths = list(np.array_split(self.paths, worker_info.num_workers)[worker_info.id])
        return PathRecordIterator(paths)

    def __getitem__(self, index) -> T_co:
        start_offset = index * record_dtype.itemsize
        so_far = 0
        for path in self.paths:
            path_size = self.path_sizes[path]
            so_far += path_size
            if so_far > start_offset:
                file_offset = start_offset - (so_far - path_size)
                with open(path, 'rb') as fd:
                    fd.seek(file_offset, io.SEEK_SET)
                    data = fd.read(record_dtype.itemsize)
                    return PathRecordIterator.to_record(data)
        raise IndexError()


class EvaluationModel(pl.LightningModule):
    def __init__(
            self,
            train_data: List[pathlib.Path],
            val_data: List[pathlib.Path],
            batch_size=1024,
            learning_rate=1e-3,
            hidden_layers=10,
            hidden_layer_width=256
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="binary")
        self.save_hyperparameters({
            'batch_size': batch_size,
            'learning_rate': learning_rate
        })

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, dtype=torch_dtype)
        #self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, dtype=torch_dtype)
        self.pool = nn.MaxPool2d(2)

        layers: List[Tuple[str, any]] = [
            (f'linear-entry', nn.Linear((64 * 3 * 3) + (12 * 8 * 8) + 3, hidden_layer_width, dtype=torch_dtype)),
            (f'relu-entry', nn.ReLU())
        ]

        for i in range(hidden_layers):
            layers.append(
                (f'linear-{i}', nn.Linear(hidden_layer_width, hidden_layer_width, dtype=torch_dtype))
            )
            layers.append(
                (f'relu-{i}', nn.ReLU())
            )
            layers.append(
                (f'dropout-{i}', nn.Dropout(0.1))
            )

        layers.append(('linear', nn.Linear(hidden_layer_width, 1, dtype=torch_dtype)))
        self.seq = nn.Sequential(collections.OrderedDict(layers))
        #self.accuracy = BinaryAccuracy()

    def forward(self, board, features):
        #flat_board = torch.flatten(board, 1)
        conv_board = self.pool(F.relu(self.conv1(board)))
        #board = self.pool(F.relu(self.conv2(board)))
        #conv_board = torch.flatten(conv_board, 1)
        x = torch.cat([
            torch.flatten(conv_board, 1),
            torch.flatten(board, 1),
            features
        ], 1)
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        y = torch.clip(batch['score'], -15, 15)
        y_hat = self.forward(batch['board'], batch['features'])
        loss = F.l1_loss(y_hat, y)
        #mse_loss = F.mse_loss(y_hat, y)
        #acc = self.accuracy(y_hat, y)
        #acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        #self.log("train_loss_mse", mse_loss)
        #self.log("train_accuracy", acc)
        return loss

    def train_dataloader(self) -> DataLoader:
        dataset = EvaluationDataset(self.train_data)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=32, pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if not self.val_data:
            return []
        dataset = EvaluationDataset(self.val_data)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=32, pin_memory=True, drop_last=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    train_paths = list(pathlib.Path('d:/chess_split_jan').rglob('shuffled*.bin'))
    val_paths = list(pathlib.Path('d:/chess_split_feb').rglob('shuffled*.bin'))
    print(f"Have {len(train_paths)} train_paths")
    old_lr = 1.7378008287493761e-06
    new_lr = 0.8317637711026709
    new_lr = 1e-3
    model = EvaluationModel(train_paths, val_paths, batch_size=4096, learning_rate=new_lr, hidden_layers=6, hidden_layer_width=2048)

    trainer = pl.Trainer(accelerator="gpu", max_epochs=1000, callbacks=[StochasticWeightAveraging(swa_lrs=1e-3)], accumulate_grad_batches=7)
    #tuner = Tuner(trainer)

    #lr_finder = tuner.lr_find(model, early_stop_threshold=None)
    #print(lr_finder.results)
    #print(lr_finder.suggestion())
    #
    # model.learning_rate = lr_finder.suggestion()
    # print("Learning rate", model.learning_rate)
    # batch_size = tuner.scale_batch_size(model, mode="power")
    # model.batch_size = batch_size
    # print("Batch size", model.batch_size)
    # Auto-scale batch size by growing it exponentially (default)

    trainer.fit(model)


if __name__ == "__main__":
    main()
