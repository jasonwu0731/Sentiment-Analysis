from keras.models import load_model
from model import SentimentLSTM
import tensorflow as tf
import numpy as np
from keras import backend as K
import os
import cPickle
from data_utils import sent2idx

tf.flags.DEFINE_string("modeldir", "./model/", "model directory")
tf.flags.DEFINE_string("modelname", 'model.h5', "model name")
tf.flags.DEFINE_integer("vocab_size", 50000, "Random state")
tf.flags.DEFINE_integer("max_len", 30, "max len of a sentence")
class_name = ['neg', 'pos']

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.ERROR)

K.set_learning_phase(1)

# Loading dictionary and model
with open('dictionary.p','rb') as f:
	[word2idx, idx2word, max_len_true] = cPickle.load(f)
model = load_model(FLAGS.modeldir+FLAGS.modelname)

# Running loop to let user input sentence
print("Trained-Model Loaded ...")
while(1):
	sent = raw_input("Enter a sentence:")
	sent_idx = sent2idx(sent, word2idx,FLAGS.max_len)
	pred_p = model.predict(sent_idx, batch_size=1)
	pred = np.argmax(pred_p, axis=1)
	print("### Predict Sentiment: %s !, with probability %s" % (class_name[pred[0]], str(pred_p[0])))
