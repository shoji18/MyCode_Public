"""
Loss functions.
You can use losses like; 
---
from losses import focal_loss

model = build_model()
model.compile(optimizer=OPTIMIZER, loss=focal_loss())
---
"""

from tensorflow as tf
import tensorflow.keras.backend as K

"""
Focal loss
ref.
  Lin et al., “Focal lossfor dense object detection,” 
  in The IEEE International Conference onComputer Vision (ICCV), Oct 2017
※ This focal_loss is for binary classification.
"""

def focal_loss(alpha=0.5, gamma=1.):

    def focal_loss_fixed(y_true, y_pred):
    
        # if y_true is 0, pt_1 will be 1, pt_0 will be y_pred,
        # and only the second item in the returned loss will remain.
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        focal_loss = - K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) \
                     + (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return focal_loss

    return focal_loss_fixed

"""
Class-balanced focal loss
ref.
  Cui et al., “Class-balancedloss based on effective number of samples,” 
  in Proceedings of the IEEE / CVF Conference 
  on Computer Vision and Pattern Recognition(CVPR), June 2019]
"""

def cb_focal_loss(n_pos, n_neg, beta=0.9999, gamma=1.):
    
    def cb_focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        focal_loss = K.pow(1. - pt_1, gamma) * K.log(pt_1) \
                     + K.pow(pt_0, gamma) * K.log(1. - pt_0)

        inv_enum_p = (1 - beta) / (1 - tf.pow(beta, n_pos))
        inv_enum_n = (1 - beta) / (1 - tf.pow(beta, n_neg))

        weights = tf.where(tf.equal(y_true, 1),
                           inv_enum_p*tf.ones_like(y_pred),
                           inv_enum_n*tf.ones_like(y_pred))

        return - K.mean(weights * focal_loss)
    
    return cb_focal_loss_fixed
