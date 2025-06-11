from keras.callbacks import Callback
import numpy as np

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0, patience=0):
        super(EarlyStoppingByLossVal, self).__init__()
        self.monitor = monitor
        self.value = value
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.value:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

class MetricLogger(Callback):
    def __init__(self):
        super(MetricLogger, self).__init__()
        self.metrics = []

    def on_epoch_end(self, epoch, logs=None):
        self.metrics.append(logs)

    def get_metrics(self):
        return np.array(self.metrics)