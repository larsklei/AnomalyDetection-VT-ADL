import tensorflow as tf
import keras_core as keras
import numpy as np

from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.image import ssim
from skimage.metrics import structural_similarity as sk_ssim

@keras.saving.register_keras_serializable()
class StructuralSimilarityIndexMeasure(keras.losses.Loss):
	"""

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
	def __init__(
			self,
			lambda1: float = 5,
			lambda2: float = 0.5,
			name: str = "CustomVisionLoss"
	):
		super().__init__(name=name)
		self.lambda1 = lambda1
		self.lambda2 = lambda2
	
	def call(self, y_true, y_pred):
		mse = MeanSquaredError()
		ssim = StructuralSimilarityIndexMeasure()
		return self.lambda1 * mse(y_true, y_pred) + self.lambda2 * ssim(y_true, y_pred)
	
	def get_config(self):
		return {
			"lambda1": self.lambda1,
			"lambda2": self.lambda2,
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
