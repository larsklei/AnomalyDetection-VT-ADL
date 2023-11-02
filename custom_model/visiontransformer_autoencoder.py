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
	"""
	def __init__(self, encoder, decoder, latent_dim, patch_size):
		self.latent_dim = latent_dim
		self.patch_size = patch_size
		self.encoder = encoder
		self.decoder = decoder

	def build(self, input_shape):
		batch_size, img_height, img_width, channels = input_shape

		self.embedding = PatchEmbedding(self.latent_dim, self.patch_size, img_height, img_width, name='Embedding')

		if self.encoder == None:
			pass

		if self.decoder == None:
			pass

	def train_step(self, images):
		with tf.GradientTape() as tape:
			patches = images
			embedded_patches = self.embedding(patches)
			encoded_patches = self.encoder(patches)
			reconstruction = self.decoder(encoded_patches)

			reconstruction_loss = VisionLoss(data, reconstruction)
			grads = tape.gradient(reconstruction_loss, self.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
			self.reconstruction_loss_tracker.update_state(reconstruction_loss)

			return {
				'reconstruction_loss': self.reconstruction_loss_tracker.result()
			}
