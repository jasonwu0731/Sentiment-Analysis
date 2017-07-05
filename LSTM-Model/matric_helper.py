from sklearn.metrics import classification_report
from keras.callbacks import Callback
import numpy as np

class ClassificationReport(Callback):
    def __init__(self, model, x_eval, y_eval, labels):
        self.model = model
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        print("Generating Classification Report:")
        pred = np.argmax(self.model.predict(self.x_eval), axis=1)
        truth = np.argmax(self.y_eval, axis=1)
        target_names = [self.labels[i] for i in range(len(self.labels))]
        print(classification_report(truth, pred, target_names=target_names))
