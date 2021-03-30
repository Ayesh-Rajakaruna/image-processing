{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "noted-boring",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras \n",
    "from tensorflow.keras import datasets, layers, models\n",
    "import matplotlib.pyplot as plt\n",
    "#from tensorflow import keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "linear-willow",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()\n",
    "# Normalize pixel values to be between 0 and 1\n",
    "x_train, x_test = x_train / 255.0, x_test / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "liked-caribbean",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = models.Sequential()\n",
    "model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))\n",
    "model.add(layers.MaxPooling2D((2, 2)))\n",
    "model.add(layers.Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(layers.MaxPooling2D((2, 2)))\n",
    "model.add(layers.Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(layers.MaxPooling2D((2, 2)))\n",
    "model.add(layers.Flatten())\n",
    "model.add(layers.Dense(64, activation='relu'))\n",
    "model.add(layers.Dense(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "eligible-sheep",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_6 (Conv2D)            (None, 30, 30, 32)        896       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_6 (MaxPooling2 (None, 15, 15, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_7 (Conv2D)            (None, 13, 13, 64)        18496     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_7 (MaxPooling2 (None, 6, 6, 64)          0         \n",
      "_________________________________________________________________\n",
      "conv2d_8 (Conv2D)            (None, 4, 4, 64)          36928     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_8 (MaxPooling2 (None, 2, 2, 64)          0         \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 256)               0         \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 64)                16448     \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 10)                650       \n",
      "=================================================================\n",
      "Total params: 73,418\n",
      "Trainable params: 73,418\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "intellectual-trunk",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_categorical = keras.utils.to_categorical(\n",
    "    y_train, num_classes=10, dtype = \"float32\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "infrared-responsibility",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test_categorical = keras.utils.to_categorical(\n",
    "    y_test, num_classes=10, dtype = \"float32\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "loved-costa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1563/1563 [==============================] - 44s 25ms/step - loss: 1.9198 - accuracy: 0.2829 - val_loss: 1.4981 - val_accuracy: 0.4716\n",
      "Epoch 2/10\n",
      "1563/1563 [==============================] - 37s 24ms/step - loss: 1.3058 - accuracy: 0.5322 - val_loss: 1.2361 - val_accuracy: 0.5539\n",
      "Epoch 3/10\n",
      "1563/1563 [==============================] - 36s 23ms/step - loss: 1.1528 - accuracy: 0.5924 - val_loss: 1.0946 - val_accuracy: 0.6078\n",
      "Epoch 4/10\n",
      "1563/1563 [==============================] - 36s 23ms/step - loss: 1.0309 - accuracy: 0.6369 - val_loss: 1.0838 - val_accuracy: 0.6157\n",
      "Epoch 5/10\n",
      "1563/1563 [==============================] - 36s 23ms/step - loss: 0.9524 - accuracy: 0.6670 - val_loss: 1.0446 - val_accuracy: 0.6385\n",
      "Epoch 6/10\n",
      "1563/1563 [==============================] - 37s 23ms/step - loss: 0.8747 - accuracy: 0.6973 - val_loss: 1.1047 - val_accuracy: 0.6207\n",
      "Epoch 7/10\n",
      "1563/1563 [==============================] - 36s 23ms/step - loss: 0.8562 - accuracy: 0.6996 - val_loss: 1.0514 - val_accuracy: 0.6502\n",
      "Epoch 8/10\n",
      "1563/1563 [==============================] - 37s 24ms/step - loss: 0.8158 - accuracy: 0.7149 - val_loss: 1.0038 - val_accuracy: 0.6644\n",
      "Epoch 9/10\n",
      "1563/1563 [==============================] - 37s 24ms/step - loss: 0.7765 - accuracy: 0.7302 - val_loss: 1.0423 - val_accuracy: 0.6499\n",
      "Epoch 10/10\n",
      "1563/1563 [==============================] - 37s 24ms/step - loss: 0.7442 - accuracy: 0.7428 - val_loss: 1.0689 - val_accuracy: 0.6576\n"
     ]
    }
   ],
   "source": [
    "opt = tf.keras.optimizers.SGD(learning_rate=1.4e-2, momentum=0.9)\n",
    "model.compile(optimizer=opt,\n",
    "              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),\n",
    "              metrics=['accuracy'])\n",
    "history = model.fit(x_train, y_train_categorical, epochs=10,\n",
    "                   validation_data=(x_test, y_test_categorical))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "conscious-pledge",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 - 2s - loss: 1.0689 - accuracy: 0.6576\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['accuracy'], label='accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label = 'val_accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.ylim([0.5, 1])\n",
    "plt.legend(loc='lower right')\n",
    "test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "honey-world",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.3848400115966797, 0.5473600029945374, 0.6039199829101562, 0.635640025138855, 0.6652799844741821, 0.6898999810218811, 0.6998599767684937, 0.7107999920845032, 0.7237799763679504, 0.7333999872207642]\n",
      "[0.4715999960899353, 0.5539000034332275, 0.6078000068664551, 0.6157000064849854, 0.6384999752044678, 0.6207000017166138, 0.6502000093460083, 0.6643999814987183, 0.6499000191688538, 0.6575999855995178]\n"
     ]
    }
   ],
   "source": [
    "print(history.history['accuracy'])\n",
    "print(history.history['val_accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fiscal-tutorial",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
