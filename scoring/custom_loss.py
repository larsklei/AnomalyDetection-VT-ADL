import tensorflow as tf
import keras_core as keras

from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.image import ssim


def StructuralSimilarityIndex(y_true, y_pred):
	r"""
	A wrapper for the ssim function as a Keras Loss function.
	"""
	ssim_value = ssim(y_true, y_pred, max_val=1.0)
	loss = 1 - tf.reduce_mean(ssim_value)

	return loss


class VisionLoss(Loss):
	r"""


	Args:
		lambda1:
		lambda2:
	"""

	def __init__(self, lambda1=5, lambda2=0.5, name=None):
		super().__init__(VisionLoss, name=name)
		self.lambda1 = lambda1
		self.lambda2 = lambda2

	def call(self, y_true, y_pred):
		mse = MeanSquaredError()
		ssim_loss = StructuralSimilarityIndex()

		return self.lambda1 * mse(y_true, y_pred) + self.lambda2 * ssim_loss(y_true, y_pred)