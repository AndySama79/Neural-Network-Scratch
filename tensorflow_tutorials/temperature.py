# %%    Imports
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# %%    Synthetic data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius):
    print(f"{c} degrees Celsius = {fahrenheit[i]} degrees Fahrenheit")

# %%    Create the model
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# assemble the layers
model = tf.keras.Sequential([l0])

# %%    Compile the model, with loss and optimzier functions
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
# %%   Train the model
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

print("Finished training the model")

# %%    Display training statistics
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

# %%   Use the model to predict values
print(model.predict([100.0]))

# %%   Looking at the layer weights
print("These are the layer variables: {}".format(l0.get_weights()))

# %%   A little experiment
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print(f"Model predicts that 100 degrees Celsius is: {model.predict([100.0])} degrees Fahrenheit")
print(f"These are the l0 variables: {l0.get_weights()}")
print(f"These are the l1 variables: {l1.get_weights()}")
print(f"These are the l2 variables: {l2.get_weights()}")
