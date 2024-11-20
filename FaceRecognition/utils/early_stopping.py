import torch
import numpy as np


class EarlyStopping:
    """Early stops the training when a metric has stopped improving"""

    def __init__(
        self,
        metric_name="Val Loss",
        latency=0,
        patience=15,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        mode="min",
    ):
        """
        Args:
            metric_name (str): the tracked metric name (e.g. validation loss)
            latency (int): After which epoch start evaluating early stopping.
                            Default: 10
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            mode (str): In 'min' mode, the training will be stopped when the quantity monitored has stopped decreasing;
                        In 'max' mode it will be reduced when the quantity monitored has stopped increasing;
                            Default: 'min'
        """
        self.latency = latency
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.starting = 0
        self.best_epoch = 0
        self.train_loss_min = np.Inf
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.last_metric_value = np.Inf
        self.accuracy_train = {}
        self.accuracy_validation = {}
        self.confusion_train = np.zeros((0, 0))
        self.confusion_val = np.zeros((0, 0))
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric_name = metric_name
        self.mode = mode

    def __call__(
        self,
        metric_value,
        model,
        epoch,
        train_loss=None,
        accuracy_train=None,
        accuracy_validation=None,
        confusion_train=None,
        confusion_val=None,
    ):
        if self.mode == "min":
            score = -metric_value
        else:
            score = metric_value
        self.starting += 1
        if self.starting >= self.latency:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.best_model = model
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_model = model
                self.best_epoch = epoch
                self.train_loss_min = train_loss
                self.accuracy_train = accuracy_train
                self.accuracy_validation = accuracy_validation
                self.save_checkpoint(score, model)
                self.counter = 0
                self.confusion_train = confusion_train
                self.confusion_val = confusion_val

    def save_checkpoint(self, metric_value, model):
        """Saves model when the monitored quantity decrease (min mode) or increase (max mode)"""
        if self.verbose:
            if self.mode == "min":
                self.trace_func(
                    f"{self.metric_name} decreased ({self.last_metric_value:.6f} --> {metric_value:.6f}. Saving model "
                    f"... "
                )
            else:
                self.trace_func(
                    f"{self.metric_name} increased ({self.last_metric_value:.6f} --> {metric_value:.6f}. Saving model "
                    f"... "
                )
        torch.save(model.state_dict(), self.path)
        self.last_metric_value = metric_value
