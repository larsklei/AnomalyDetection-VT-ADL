import tensorflow as tf
import keras_core as keras
import numpy as np

from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.image import ssim
from skimage.metrics import structural_similarity as sk_ssim

def gaussian_filter(filter_size: int, sigma: float):
	"""Returns a Gaussian filter of filter size and variance sigma.
	
	Args:
		filter_size: int
			The size of the filter.
		sigma: float
			The variance for the filter. It has to be positive (>0).

	Returns:
		A tensor of the shape (filter_size, filter_size) with values defined by the Gaussian filter.

	"""
	filter_size = keras.ops.convert_to_tensor(filter_size, dtype=tf.int32)
	sigma = keras.ops.convert_to_tensor(sigma, dtype=tf.float32)
	
	if sigma <= 0:
		raise ValueError("The variance sigma has to be positive.")
	
	filter = keras.ops.arange(filter_size, dtype=sigma.dtype)
	filter = keras.ops.square(filter)
	filter *= -0.5 / keras.ops.square(sigma)
	
	filter = keras.ops.reshape(filter, new_shape=[1, -1]) + keras.ops.reshape(filter, new_shape=[-1, 1])
	filter = keras.ops.reshape(filter, new_shape=[1, -1])
	
	filter = keras.ops.softmax(filter)
	
	return keras.ops.reshape(filter, new_shape=[filter_size, filter_size])

@keras.saving.register_keras_serializable()
class StructuralSimilarityIndexMeasure(keras.losses.Loss):
	"""Wrapper to use the tf.image function as a loss function.
	
	Args:
		max_val:
		name: str
			Name of the loss function.
	"""
	def __init__(
			self,
			max_val: int = 1,
			name: str = "ssim"
	):
		super().__init__(name=name)
		self.max_val = max_val
	
	def call(self, y_true, y_pred):
		return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=self.max_val))
	
	def get_config(self):
		pass


@keras.saving.register_keras_serializable()
class VisionLoss(keras.losses.Loss):
	"""A loss function which is the sum of the MSE and the SSIM loss.
	
	Attributes
		lambda1: float
			The weight of the MSE term.
		lambda2: float
			The weight of the SSIM term.
		name:
	"""
	def __init__(
			self,
			lambda_mse: float = 5,
			lambda_ssim: float = 0.5,
			name: str = "CustomVisionLoss"
	):
		super().__init__(name=name)
		self.lambda_mse = lambda_mse
		self.lambda_ssim = lambda_ssim
	
	def call(self, y_true, y_pred):
		mse = MeanSquaredError()
		ssim = StructuralSimilarityIndexMeasure()
		return self.lambda_mse * mse(y_true, y_pred) + self.lambda_ssim * ssim(y_true, y_pred)
	
	def get_config(self):
		return {
			"lambda_mse": self.lambda_mse,
			"lambda_ssim": self.lambda_ssim,
			"name": self.name
		}


def get_ground_truth_predictions(y_true, y_pred, threshold):
	y_true_np = np.squeeze(y_true.numpy())
	y_pred_np = np.squeeze(y_pred.numpy())
	_, ssim_residual_mask = sk_ssim(
		y_true_np,
		y_pred_np,
		data_range=1,
		channel_axis=-1,
		gaussian_weights=True,
		full=True,
		use_sample_covariance=False,
		sigma=1.5
	)
	gt_pred = np.zeros(shape=ssim_residual_mask.shape)
	gt_pred[(1 - ssim_residual_mask) > threshold] = 1
	gt_pred = tf.convert_to_tensor(gt_pred, dtype=tf.float32)
	return gt_pred