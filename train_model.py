from process_img_to_npz import process

import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# kegai
IMAGE_SOURCE = '8'
MOVE_CATEGORIES = 3
RESOLUTION_LEVEL = 1/2
GRAY_IMG = False

process(IMAGE_SOURCE, MOVE_CATEGORIES, RESOLUTION_LEVEL, GRAY_IMG)

if GRAY_IMG:
    IMAGE_CHANNELS = 1
else:
    IMAGE_CHANNELS = 3
IMAGE_HEIGHT, IMAGE_WIDTH = int(120*RESOLUTION_LEVEL), int(160*RESOLUTION_LEVEL)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
NORMALIZATION = True


def load_data():
	""" Load training data and split it into training and validation set """

	# load training data
	image_array = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
	label_array = np.zeros((1, MOVE_CATEGORIES), 'float')
	training_data = glob.glob('../DATA/training_data_npz/*.npz')

	# if no data, exit
	if not training_data:
		print("No training data in directory, exit")
		sys.exit()

	for single_npz in training_data:
		with np.load(single_npz) as data:
			train_temp = data['train_images']
			train_labels_temp = data['train_labels']
		image_array = np.vstack((image_array, train_temp))
		label_array = np.vstack((label_array, train_labels_temp))

	X = image_array[1:, :]
	y = label_array[1:, :]

	print('Image array shape: ' + str(X.shape))
	print('Label array shape: ' + str(y.shape))
	print('Image array mean: ', np.mean(X))
	print('Image array var: ', np.var(X))

	# now we can split the data into a training (80), testing(20), and validation set thanks to scikit learn
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

	return X_train, X_valid, y_train, y_valid

x_train, x_test, y_train, y_test = load_data()


print("Samples for trainng: ", x_train.shape[0])
print("Samples for validation:", x_test.shape[0])
plt.figure()
if GRAY_IMG:
    plt.imshow(x_test[3].reshape(IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.uint8))
else:
    plt.imshow(x_test[120].astype(np.uint8))
plt.show()
print(y_test[3])

if NORMALIZATION:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


# step5
model = Sequential()
model.add(BatchNormalization(input_shape=INPUT_SHAPE))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=INPUT_SHAPE))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(MOVE_CATEGORIES, activation='softmax'))


learning_rate = 0.0001
model.compile(optimizer=Adam(lr=learning_rate), # tf.keras.optimizers.Adadelta(),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.0005, patience=5, verbose=1, mode='min')

import time
start_time = time.time()

epochs = 1
batch_size_per_step = 20

model.fit(x_train, y_train,
          batch_size=batch_size_per_step,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test),
          callbacks=[early_stop])

end_time = time.time()

# step8
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print("Training spends {}s".format(end_time-start_time))

# step9
prediction = model.predict(x_test[3].reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
print(prediction)
prediction = prediction.argmax()

plt.figure()

if NORMALIZATION:
    x_test *= 255
if GRAY_IMG:
    plt.imshow(x_test[3].reshape(IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.uint8))
else:
    plt.imshow(x_test[3].astype(np.uint8))
plt.text(0, -3, prediction)
plt.show()

model_name = "car_model_from_source_"+IMAGE_SOURCE+".h5"
model.save(model_name)

from keras.models import load_model

print("loading model...")
model = load_model("car_model_from_source_{}.h5".format(IMAGE_SOURCE))
print("model loaded")
