import pathlib

import torch
import torch.nn.functional as F

from train4 import EvaluationModel, Converter


def main():
    paths = list(pathlib.Path("d:/chess/split/jan").rglob("shuffled-1.bin"))
    print(f"Have {len(paths)} paths")

    # trainer = pl.Trainer(gpus=1)
    model_path = sorted(pathlib.Path("logs_train4/lightning_logs/version_0/checkpoints").rglob("*.ckpt"))[-1]
    print(f"Loading {model_path}")
    model = EvaluationModel.load_from_checkpoint(model_path)
    # trainer.test(model)
    model.eval()
    model.freeze()
    model.validation_file_count = 0
    loader = model.train_dataloader()
    min_errors = []
    max_errors = []
    mean_errors = []
    try:
        for sample in loader:
            y = sample["score"]
            y_hat = model.forward(sample["boards"])

            loss = F.l1_loss(y_hat, y)
            error = torch.abs(y_hat - y)
            min_error = torch.min(error)
            min_errors.append(min_error)
            max_error = torch.max(error)
            max_errors.append(max_error)
            mean_error = torch.mean(error)
            mean_errors.append(mean_error)
            print(f"Loss: {loss:.2f} {min_error:.2f}/{max_error:.2f}/{mean_error:.2f}")
            max_error_index = torch.argmax(error)
            record = Converter.tensor_record_to_record(sample["boards"][max_error_index])
            board = Converter.record_to_board(record)
            print(board)
            print(y[max_error_index], y_hat[max_error_index])
    finally:
        print(
            f"Final: {min(min_errors):.2f}/{max(min_errors):.2f} {max(max_errors):.2f}/{min(max_errors):.2f} {sum(mean_errors) / len(mean_errors):.2f}"
        )
        # ymean = sum(ys) / len(ys)
        # sst = sum([
        #     (y - ymean)**2
        #     for y in ys
        # ])
        # ssr = sum([
        #     (y - yh)**2
        #     for y, yh in zip(ys, yhats)
        # ])
        # print(f"R2 = {float(1) - (ssr/sst)}")


if __name__ == "__main__":
    main()
