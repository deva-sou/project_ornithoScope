from tensorflow.keras.callbacks import Callback

class AntiNaN(Callback):
    """
    Stop training when loss reach NaN.
    """

    def __init__(self):
        super(AntiNaN, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss and not loss > 0:
            self.model.stop_training = True
            print("\nNaN stopping because loss diverged.")