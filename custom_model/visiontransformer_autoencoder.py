import tensorflow as tf
import keras_core as keras

from custom_model.vision_transformer import PatchEmbedding
from scoring.custom_loss import VisionLoss

class VisionTransformerEncoder(keras.Model):
	r"""
	Args:
		encoder:
		decoder:
		latent_dim:
		patch_size:
		num_register:
	"""
	def __init__(self, emb_dim, patch_size, encoder=None, decoder=None, num_register=None):
		self.embedding = None
		self.emb_dim = emb_dim
		self.patch_size = patch_size
		self.encoder = encoder
		self.decoder = decoder
		self.num_register = num_register

	def build(self, input_shape):
		batch_size, img_height, img_width, channels = input_shape

		self.embedding = PatchEmbedding(self.emb_dim, self.patch_size, img_height, img_width, name='Embedding')

		if self.encoder is None:
			pass

		if self.decoder is None:
			pass

	def train_step(self, images):
		batch_size = images.shape[0]

		with tf.GradientTape() as tape:
			patches = images
			embedded_patches = self.embedding(patches)

			if self.num_register is not None:
				register_tokens = tf.random.normal(shape=[batch_size, self.num_register, self.emb_dim])
				embedded_patches = tf.concat(embedded_patches, register_tokens)

			encoded_patches = self.encoder(embedded_patches)

			encoded_patches_wo_registers = encoded_patches[:, :-self.num_register, :]

			representation = keras.layers.LayerNormalization()(encoded_patches_wo_registers)
			representation = keras.layers.Flatten()(representation)
			representation = keras.layers.Dropout(0.5)(representation)

			reconstruction = self.decoder(representation)

			reconstruction_loss = VisionLoss(images, reconstruction)
			grads = tape.gradient(reconstruction_loss, self.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
			self.reconstruction_loss_tracker.update_state(reconstruction_loss)

			return {
				'reconstruction_loss': self.reconstruction_loss_tracker.result()
			}
