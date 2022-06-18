# Acharya model
# This model needs large input size

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Input,
                                     MaxPool1D, LeakyReLU, Dense, Flatten)

#lossはおそらくcategorical_crossentropy
def build_model(n_channels=1, len_seq=4097):
    input = Input(shape=(len_seq, n_channels))
    x = Conv1D(filters=4, kernel_size=6, strides=1, padding="valid")(input)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Conv1D(filters=4, kernel_size=5, strides=1, padding="valid")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Conv1D(filters=10, kernel_size=4, strides=1, padding="valid")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Conv1D(filters=10, kernel_size=4, strides=1, padding="valid")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Conv1D(filters=15, kernel_size=4, strides=1, padding="valid")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(50)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(20)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(3, activation="softmax")(x)

    return Model(input, x)


if __name__ == '__main__':

    build_model().summary()