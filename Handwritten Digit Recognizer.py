import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("train size:", x_train.shape)
print("test size:", x_test.shape)

x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # softmax for multi-class output

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# training for 10 epochs, batch size 128 works fine here
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1)

# evaluate on test data
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

# plot how accuracy and loss changed over training
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='train')
axes[0].plot(history.history['val_accuracy'], label='val')
axes[0].set_title('Accuracy per epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss per epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

preds = model.predict(x_test[:10])
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    col = 'green' if pred_labels[i] == true_labels[i] else 'red'
    plt.title(f"P:{pred_labels[i]}\nT:{true_labels[i]}", fontsize=7, color=col)
    plt.axis('off')

plt.suptitle("Predictions  |  Green = correct  Red = wrong")
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()

model.save('mnist_model.h5')
print("saved model to mnist_model.h5")

