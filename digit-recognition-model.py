import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

path = 'https://assets.datacamp.com/production/course_1975/datasets/mnist.csv'

data = pd.read_csv(path)

target = to_categorical(data['5'])
predictors = data.drop('5', axis=1)

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, validation_split=0.3)
