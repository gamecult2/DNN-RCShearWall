import torch
import numpy as np

class EarlyStopping:
    """Early stopping mechanism for PyTorch training.

    Args:
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 7.
        verbose (bool, optional): Whether to print messages during training. Defaults to False.
        delta (float, optional): Minimum change in the monitored score to consider it an improvement. Defaults to 0.
        path (str, optional): Path to save the model checkpoint. Defaults to 'checkpoint.pt'.
        monitor (str, optional): Metric to monitor for early stopping (e.g., 'val_loss', 'val_acc'). Defaults to 'val_loss'.
        mode (str, optional): Mode for monitoring the metric ('min' or 'max'). Defaults to 'min'.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss