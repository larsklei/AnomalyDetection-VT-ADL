import tensorflow as tf
import keras_core as keras

from tensorflow.image import ssim

from vit_ae.custom_model.visiontransformer_layers import PatchEmbedding, VisionTransformerBlock, get_decoder
from vit_ae.model_utilities.loss_metric_utilities import VisionLoss


@keras.saving.register_keras_serializable()
class VisionTransformerAutoencoder(keras.Model):
	"""

	Args:
		embed_dim: int
		
		patch_sizes: int
		
		encoder: keras.Model
		
		decoder: keras.Model
		
		img_height: int
		
		img_width: int
		
		num_register: int
	"""
	def __init__(
			self,
			embed_dim: int,
			patch_size: int,
			encoder: keras.Model,
			decoder: keras.Model,
			img_height: int,
			img_width: int,
			num_register: int | None = None,
			threshold: float = 0.2,
			name: str | None = None,
			**kwargs
	):
		if (img_height % patch_size != 0) or (img_width % patch_size != 0):
			raise ValueError("Image height or width is not divisible by patch_size.")
		super().__init__(name=name, **kwargs)
		self.embed_dim = embed_dim
		self.patch_size = patch_size
		self.encoder = encoder
		self.decoder = decoder
		self.img_height = img_height
		self.img_width = img_width
		self.num_register = num_register
		self.threshold = threshold
		self.num_patches_root = tf.cast(img_height / self.patch_size, "int32")
		self.embedding = PatchEmbedding(
			self.embed_dim,
			self.patch_size,
			self.img_height,
			self.img_width,
			name="Embedding"
		)
	
		self.latent_layers = [
			keras.layers.LayerNormalization(),
			keras.layers.Reshape(
				target_shape=(self.num_patches_root, self.num_patches_root, self.embed_dim)
			)
		]

		self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
		self.ssim_gt_tracker = keras.metrics.Mean(name="groundtruth_L1")
		
	def call(self, inputs, training=False):
		batch_size = inputs.shape[0]
		
		patches = inputs
		embedded_patches = self.embedding(patches)
		
		if self.num_register is not None:
			register_tokens = tf.random.normal(shape=(batch_size, self.num_register, self.embed_dim))
			embedded_patches = tf.concat([embedded_patches, register_tokens], axis=1)
			encoded_patches = self.encoder(embedded_patches)
			representation = encoded_patches[:, :-self.num_register, :]
		else:
			representation = self.encoder(embedded_patches)
			
		for layer in self.latent_layers:
			representation = layer(representation)
		
		reconstructed = self.decoder(representation)
		return reconstructed
	
	@property
	def metrics(self):
		return [
			self.reconstruction_loss_tracker,
			self.ssim_gt_tracker
		]
	
	@tf.function
	def compute_mask_L1(self, images, reconstructions, threshold):
		diff = tf.math.abs(images-reconstructions)
		threshold_tensor = threshold * tf.ones(shape=images.shape)
		masks = tf.cast(tf.math.greater(threshold_tensor, diff), dtype=tf.float32)
		
		return masks

	@tf.function
	def train_step(self, data):
		images, ground_truths = data
		with tf.GradientTape() as tape:
			reconstructions = self(images, training=True)
			
			if reconstructions.shape[1:] != images.shape[1:]:
				raise ValueError("Decoder is wrong. It does not reproduce the correct shape.")
			reconstruction_loss = VisionLoss()(images, reconstructions)
		trainable_vars = self.trainable_variables
		grads = tape.gradient(reconstruction_loss, trainable_vars)
		self.optimizer.apply_gradients(zip(grads, trainable_vars))
		diff = tf.math.abs(images - reconstructions)
		threshold_tensor = self.threshold * tf.ones(shape=images.shape)
		masks = tf.cast(tf.math.greater(threshold_tensor, diff), dtype=tf.float32)
		ssim_value = 1-ssim(ground_truths, masks, max_val=1)
		
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.ssim_gt_tracker.update_state(ssim_value)
		
		return {m.name: m.result() for m in self.metrics}
		
	@tf.function
	def test_step(self, data):
		images, ground_truth = data
		reconstructions = self(images, training=False)
		reconstruction_loss = VisionLoss()(images, reconstructions)
		diff = tf.math.abs(images - reconstructions)
		threshold_tensor = self.threshold * tf.ones(shape=images.shape)
		masks = tf.cast(tf.math.greater(threshold_tensor, diff), dtype=tf.float32)
		ssim_value = 1-ssim(ground_truth, masks, max_val=1)
		
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.ssim_gt_tracker.update_state(ssim_value)
		
		return {m.name: m.result() for m in self.metrics}
	
	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"embed_dim": self.embed_dim,
				"patch_size": self.patch_size,
				"encoder": self.encoder,
				"decoder": self.decoder,
				"img_height": self.img_height,
				"img_width": self.img_width,
				"num_register": self.num_register,
				"threshold": self.threshold
			}
		)
		
		return config
	
	@classmethod
	def from_config(cls, config):
		config["encoder"] = keras.saving.deserialize_keras_object(config["encoder"])
		config["decoder"] = keras.saving.deserialize_keras_object(config["decoder"])
		return cls(**config)
		
		
		
	
if __name__ == "__main__":
	decoder = get_decoder()
	encoder = keras.Sequential(
		[
			VisionTransformerBlock(num_heads=4, key_dim=512, hidden_units=32, num_outputs=512),
			VisionTransformerBlock(num_heads=4, key_dim=512, hidden_units=32, num_outputs=512)
		]
	)

	input_shape = (4, 512, 512, 3)

	ViT = VisionTransformerAutoencoder(
		embed_dim=512,
		patch_sizes=16,
		encoder=encoder,
		decoder=decoder,
		img_height=512,
		img_width=512,
		num_register=1
	)

	input_dummy = tf.random.normal(shape=input_shape)
	
	ViT.compile(
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		run_eagerly=True
	)

	ViT.fit(input_dummy, input_dummy)