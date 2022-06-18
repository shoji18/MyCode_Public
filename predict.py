import tensorflow as tf
from tensorflow.keras.models import load_model
import my_utility as myutil

tf.config.experimental.set_memory_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())

# Prediction method
# Input: [n_samples, N_CHANNELS, LEN_SEQ, 1]
# Output: [n_samples, N_CHANNELS]
def predict(modelpath, test_data):

    model = load_model(modelpath)
    pred = model.predict(test_data)

    return pred