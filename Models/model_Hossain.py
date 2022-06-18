from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Input,
                                     MaxPool2D, Dense, Flatten)

def build_model(n_channels=16, len_seq=500):
    input = Input(shape=(n_channels, len_seq, 1))
    x = Conv2D(filters=20, kernel_size=(1, 10),
               strides=1, padding="valid", activation='elu',
               data_format="channels_last")(input)
    x = Conv2D(filters=20, kernel_size=(n_channels, 1),
               strides=1, padding="valid", activation='elu',
               data_format="channels_last")(x)
    x = MaxPool2D(pool_size=(1, 2), strides=2,
                  data_format="channels_last")(x)
    x = Conv2D(filters=40, kernel_size=(1, 10),
               strides=1, padding="valid", activation='elu',
               data_format="channels_last")(x)
    x = MaxPool2D(pool_size=(1, 2), strides=2,
                  data_format="channels_last")(x)
    x = Conv2D(filters=80, kernel_size=(1, 10),
               strides=1, padding="valid", activation='elu',
               data_format="channels_last")(x)
    x = Flatten()(x)
    x = Dense(16, activation="sigmoid")(x)

    return Model(input, x)


# original model (for reference)
# activation func. of final FC layer is softmax and output dim is 2
def build_model_original(n_channels=23, len_seq=512):
    input = Input(shape=(len_seq, n_channels, 1))
    x = Conv2D(filters=20, kernel_size=(10, 1),
               strides=1, padding="valid", activation='elu')(input)
    x = Conv2D(filters=20, kernel_size=(1, 23),
               strides=1, padding="valid", activation='elu')(x)
    x = MaxPool2D(pool_size=(2, 1), strides=2)(x)
    x = Conv2D(filters=40, kernel_size=(10, 1),
               strides=1, padding="valid", activation='elu')(x)
    x = MaxPool2D(pool_size=(2, 1), strides=2)(x)
    x = Conv2D(filters=80, kernel_size=(10, 1),
               strides=1, padding="valid", activation='elu')(x)
    x = Dense(2, activation="softmax")(x)

    return Model(input, x)


if __name__ == '__main__':

    build_model().summary()