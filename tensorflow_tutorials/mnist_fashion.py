# %%    Imports and Helper Libraries
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt
# %%    Import the Fashion MNIST dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_ds, test_ds = dataset['train'], dataset['test']
# %%
class_names = metadata.features['label'].names

print(f"Class names: {class_names}")
# %%    Explore
num_train_ex = metadata.splits['train'].num_examples
num_test_ex = metadata.splits['test'].num_examples

print(f"Number of training examples: {num_train_ex}")
print(f"Number of test examples: {num_test_ex}")
# %%    Preprocess
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# using map() to apply normalize() to each element in the train and test datasets
train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

# cache() keeps them in memory after they're loaded off disk during the first epoch
train_ds = train_ds.cache()
test_ds = test_ds.cache()
# %%    Explore Processed
# remove the color dimension by reshaping
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
# %%    Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# %%    Compile the model
# using adam optimizer and sparse_categorical_crossentropy loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# %%    Train the model
BATCH_SIZE = 32
train_ds = train_ds.cache().repeat().shuffle(num_train_ex).batch(BATCH_SIZE)
test_ds = test_ds.cache().batch(BATCH_SIZE)

with tf.device('/gpu:0'):
    model.fit(train_ds, epochs=5, steps_per_epoch=math.ceil(num_train_ex/BATCH_SIZE))
# %%    Model Summary
model.summary()
# %%    Evaluate accuracy
test_loss, test_acc = model.evaluate(test_ds, steps=math.ceil(num_test_ex/BATCH_SIZE))

print('Test accuracy: ', test_acc)
# %%    Make predictions and explore
for test_images, test_labels in test_ds.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print(predictions.shape)
print(predictions[0])
print(np.argmax(predictions[0])) # image is shirt, class_names[6]
print(test_labels[0])
# %%    Graphs
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})", color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')
# %%   Verify predictions
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
# %%
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
# %%    Plot the first X test images, their predicted labels, and the true labels
# Color correct predictions in green, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
# %%    Use the trained model
# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member
img = np.array([img])
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
# %%
