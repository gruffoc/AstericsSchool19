from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
model = Sequential()

model.add(Conv2D(10, (5, 5), strides=1, padding='same',input_shape=(32, 32, 1)))
model.add(MaxPool2D(((2, 2))))
model.add(Conv2D(25, (5, 5))
model.add(MaxPool2D(((2, 2))))
model.add(Conv2D(100, (4, 4))
model.add(MaxPool2D(((2, 2))))
model.add(Flatten())
model.add(Dense(10))

model.summary()
