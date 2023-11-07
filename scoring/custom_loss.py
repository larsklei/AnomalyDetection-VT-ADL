import tensorflow as tf
import keras_core as keras

from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.image import ssim

def loss_vision(y_true, y_pred):
	ssim_value = ssim(y_true, y_pred, max_val=1.0)
	ssim_loss = 1 - tf.reduce_mean(ssim_value)

	mse_loss = MeanSquaredError()(y_true, y_pred)

	return 5*mse_loss+0.5*ssim_loss