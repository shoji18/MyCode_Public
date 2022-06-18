from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Input,
                                     AveragePooling2D, Dense, Flatten)
def build_model(n_channels=16, len_seq=500):
    input = Input(shape=(n_channels, len_seq, 1))
    x = Conv2D(6, (5, 5), strides=1, padding="same")(input)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(16, activation="sigmoid")(x)

    return Model(input, x)

if __name__ == '__main__':

    build_model().summary()