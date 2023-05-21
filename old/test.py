import pathlib

import torch
import pytorch_lightning as pl
import numpy as np

from train import EvaluationModel, EvaluationDataset

def main():
    paths = list(pathlib.Path('d:/chess_split').rglob('shuffled*.bin'))
    print(f"Have {len(paths)} paths")

    #trainer = pl.Trainer(gpus=1)
    model = EvaluationModel.load_from_checkpoint(next(pathlib.Path(
        "../lightning_logs/version_1/checkpoints").rglob("*.ckpt")))#, paths=paths, hidden_layers=10)
    #trainer.test(model)
    model.eval()
    model.freeze()
    #model.train_dataloader()
    dataset = EvaluationDataset(model.paths)
    ys = []
    yhats = []
    try:
        for sample in dataset:
            y = sample['eval'][0]
            ys.append(y)
            y_hat = model(torch.from_numpy(sample['bin']))[0].item()
            yhats.append(y_hat)
            #print(y, y_hat)
    finally:
        ymean = sum(ys) / len(ys)
        sst = sum([
            (y - ymean)**2
            for y in ys
        ])
        ssr = sum([
            (y - yh)**2
            for y, yh in zip(ys, yhats)
        ])
        print(f"R2 = {float(1) - (ssr/sst)}")




if __name__ == '__main__':
    main()