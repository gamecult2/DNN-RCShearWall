import torch
import numpy as np
import os


class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_dir='checkpoints',
                 model_name='model', save_full_model=True, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model_name = model_name
        self.save_full_model = save_full_model

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
        # Generate unique filename with timestamp

        if self.save_full_model:
            # Save entire model
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'{self.model_name}_best_full.pt'
            )
            torch.save(model, checkpoint_path)
        else:
            # Save only model state dictionary
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'{self.model_name}_best_state.pt'
            )
            torch.save(model.state_dict(), checkpoint_path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                  f'Saving model to {checkpoint_path}')

        self.val_loss_min = val_loss