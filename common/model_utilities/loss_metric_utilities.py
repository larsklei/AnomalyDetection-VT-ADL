import tensorflow as tf
import keras_core as keras

from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.image import ssim


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
	

@keras.saving.register_keras_serializable()
class IntersectionOverUnion(keras.metrics.Metric)
	def __init__(self, name="intersection_over_union"):