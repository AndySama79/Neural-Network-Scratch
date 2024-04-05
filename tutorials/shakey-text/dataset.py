import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.autograph.set_verbosity(3)


import keras
import numpy as np

def datset():
    shakespeare_url = "https://homl.info/shakespeare"
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)

    with open(filepath) as f:
        shakespeare_text = f.read()

    yield shakespeare_text

def tokenize(text):
    tokenizer = keras.preprocessing.txt.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([text])

    yield np.array(tokenizer.texts_to_sequences([text])) - 1

def split(text):
    total_length = len(text)
    train_size = total_length * 90 // 100

    dataset = tf.data.Dataset.from_tensor_slices(text[:train_size])

    return dataset