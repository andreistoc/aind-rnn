import numpy as np
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

def window_transform_series(series, window_size):
    # containers for input/output pairs

    X = []
    y = []

    for i in range(window_size, len(series)):
        X.append(series[i - window_size: i])

    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X, y


def build_part1_RNN(window_size):
    model = Sequential()

    model.add(LSTM(units = 5, input_shape = (window_size, 1)))
    model.add(Dense(1))

    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']

    unique_characters =''.join(set(text))

    print(unique_characters)

    for character in unique_characters:
        if not (character in string.ascii_lowercase or character in punctuation):
            text = text.replace(character, '')
            unique_characters = unique_characters.replace(character, '')

    print(unique_characters)

    return text


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = window_size

    while i < len(text):
        inputs.append(text[i - window_size: i])
        outputs.append(text[i])
        i += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation = 'softmax'))

    return model
