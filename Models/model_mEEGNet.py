from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm


def build_model(n_channels=16, len_seq=500, dropoutRate=0.5, kernLength=250,
                F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    
    input1 = Input(shape=(n_channels, len_seq, 1))

    block1 = Conv2D(F1, (1, kernLength), padding = 'same',
                    input_shape = (n_channels, len_seq, 1), use_bias = False,
                    data_format="channels_last")(input1)
    block1 = BatchNormalization(axis = 1)(block1)
    block1 = DepthwiseConv2D((n_channels, 1), use_bias = False, 
                             depth_multiplier = D,
                             depthwise_constraint = max_norm(1.),
                             data_format="channels_last")(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4), data_format="channels_last")(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same',
                             data_format="channels_last")(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format="channels_last")(block2)
    block2 = Dropout(dropoutRate)(block2)
        
    flatten = Flatten(name='flatten')(block2)
    
    dense = Dense(n_channels, name='dense', 
                  kernel_constraint = max_norm(norm_rate))(flatten)
    sigmoid = Activation('sigmoid', name='sigmoid')(dense)
    
    return Model(inputs=input1, outputs=sigmoid)
