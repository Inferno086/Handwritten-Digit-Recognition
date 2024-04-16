import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Loading the data"""
(trainx, trainy), (testx, testy) = tf.keras.datasets.mnist.load_data()

"""Preprocessing the data"""
trainx = trainx / 255.0
testx = testx / 255.0

"""Building the Model"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

"""Compile the Model"""
model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

"""Training"""
model.fit(trainx, trainy, epochs=4)


"""Evaluating the model"""
test_loss, test_accuracy = model.evaluate(testx, testy, verbose=1)
print(f"\n\nTest Accuracy : {test_accuracy}")


"""Making the Predictions"""

predictions = model.predict(testx)

for i in range(9999):
    print(f"\n\nActual value : {testy[i]}")
    print(f"Prediction : {np.argmax(predictions[i])}")
    f = plt.figure()
    f.set_figwidth(3)
    f.set_figheight(3)
    plt.imshow(testx[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()