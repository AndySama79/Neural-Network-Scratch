import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import keras
tf.autograph.set_verbosity(3)

# download the dataset
shakepeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakepeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()

# tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

# encode the text
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

# split the text
train_size = len(encoded) * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

# chop the text into windows
n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

datset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=39), Y_batch)
    )
dataset = dataset.prefetch(1)

# model
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, 39],
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(39, activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=100)

# use
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, depth=39)

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

print(complete_text("t", temperature=0.2))
print(complete_text("w", temperature=1))
print(complete_text("w", temperature=2))

print(complete_text("a", temperature=0.2))
print(complete_text("a", temperature=1))
print(complete_text("o", temperature=2))