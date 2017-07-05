import time
from model import SentimentLSTM
from matric_helper import ClassificationReport
from keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import os
from data_utils import build_data2train

# Input training dataset and class type
data_folder = ["../data/twitter_neg.csv", "../data/twitter_pos.csv"]
class_name = ['neg', 'pos']
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Number of batch size")
tf.flags.DEFINE_integer("nb_class", 2, "Number of classification")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_float("lstm_dropout", 0.2, "dropout_rate_for lstm")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate for optimizer")
tf.flags.DEFINE_integer("num_units", 20, "Number of units for lstm")
tf.flags.DEFINE_integer("embedding_size", 128, "Embedding size")
tf.flags.DEFINE_string("logdir", "", "log directory")
tf.flags.DEFINE_string("modeldir", "./model/", "model directory")
tf.flags.DEFINE_float("testing_ratio", 0.1, "testing ratio of the dataset")
tf.flags.DEFINE_string("random_state", 88, "Random state")
tf.flags.DEFINE_integer("vocab_size", 50000, "Random state")
tf.flags.DEFINE_integer("max_len", 30, "max len of a sentence")
tf.flags.DEFINE_boolean("is_bidirectional", False, "Whether to use bidirectional")
tf.flags.DEFINE_boolean("is_attention", False, "Whether to use attention network")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.ERROR)

assert len(data_folder) == FLAGS.nb_class
assert len(class_name) == FLAGS.nb_class

data, label, word2idx, idx2word = build_data2train(data_folder, FLAGS.vocab_size, FLAGS.max_len)
trainD, testD, trainL, testL = train_test_split(data, label, test_size=FLAGS.testing_ratio, random_state=88)

K.set_learning_phase(1)

lstm = SentimentLSTM(
                    n_classes=FLAGS.nb_class,
                    vocab_size=FLAGS.vocab_size,
                    max_len=FLAGS.max_len,
                    num_units=FLAGS.num_units,
                    useBiDirection=FLAGS.is_bidirectional,
                    useAttention=FLAGS.is_attention,
                    dropout=FLAGS.lstm_dropout,
                    learning_rate=FLAGS.learning_rate,
                    embedding_size=FLAGS.embedding_size
                    )

logdir = "./logs/"
if not os.path.exists(logdir):
    os.mkdir(logdir)
logdir += FLAGS.logdir if FLAGS.logdir else str(time.time())

print("saving logs & ckpt in %s" % logdir)

tb_callback = TensorBoard(log_dir=logdir, histogram_freq=0,
                  write_graph=True, write_images=True)

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

lstm.model.fit( trainD, trainL, 
                verbose=1, epochs=FLAGS.num_epochs,
                batch_size=FLAGS.batch_size,
                callbacks=[tb_callback, early_stop_callback],
                validation_split=0.1
            )

print("Training Finished")

if not os.path.exists(FLAGS.modeldir):
    os.mkdir(FLAGS.modeldir)
modelnameAdd = ''
if FLAGS.is_bidirectional:
    modelnameAdd += '-bidir'
if FLAGS.is_attention:
    modelnameAdd += '-att'
lstm.model.save(FLAGS.modeldir+'model'+modelnameAdd+'.h5')
print("Model Saved ...")

score = lstm.model.evaluate(testD, testL, batch_size=100, verbose=1) # Returns the loss value & metrics values for the model in test mode.
print("Testing Accuracy:", score[1])
