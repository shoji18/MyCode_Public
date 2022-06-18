import numpy as np
import glob

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Activation, Input,
                                     Dropout, Dense, Flatten, PReLU)
def build_model():
    input1 = Input((5, 4, 500))
    x = Conv2D(1024, 1, padding="same", data_format="channels_last")(input1)
    x = block(x, 512, 2)
    x = block(x, 256, 2)
    x = block(x, 128, 2)
    x = block(x, 64, 2)
    x = block(x, 32, 2)
    x = Flatten()(x)
    x = Dense(16, activation="sigmoid")(x)

    return Model(input1, x)


def block(input1, ch, reps):
    x = input1

    for _ in range(reps):
        x = PReLU()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(ch, 3, padding="same", data_format="channels_last")(x)

    return x
