import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# loading the dataset - its already inside keras so no extra download needed
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("train size:", x_train.shape)
print("test size:", x_test.shape)

# pixel values go from 0 to 255
# dividing by 255 brings everything between 0 and 1
# neural networks converge much faster with normalized inputs
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# each image is 28x28 = 784 pixels
# flattening into a 1D array so the dense layers can take it as input
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# one-hot encoding: digit 3 becomes [0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

# building the network
# went with two hidden layers, 128 neurons then 64
# added dropout in between to reduce overfitting
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

# quick visual check on 10 random test images
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

# save so i dont have to retrain every single time
model.save('mnist_model.h5')
print("saved model to mnist_model.h5")

# ---- test on your own handwritten image ----
# uncomment this block and put your image path below
# from PIL import Image
# img = Image.open("my_digit.png").convert('L').resize((28, 28))
# arr = np.array(img).astype('float32') / 255.0
# arr = arr.reshape(1, 784)
# out = model.predict(arr)
# print("my digit is:", np.argmax(out))
