import tensorflow as tf
import keras_core as keras

from custom_model.visiontransformer_layers import PatchEmbedding, VisionTransformerEncoder, get_decoder
from scoring.custom_loss import loss_vision

class VisionTransformerAutoencoder(keras.Model):
	r"""
	Args:
		encoder:
		decoder:
		latent_dim:
		patch_size:
		num_register:
	"""
	def __init__(self, latent_dim, patch_size, encoder, decoder, img_height, img_width, num_register=None):
		super().__init__()
		self.latent_dim = latent_dim
		self.patch_size = patch_size
		self.encoder = encoder
		self.decoder = decoder
		self.img_height = img_height
		self.img_width = img_width
		self.num_register = num_register
		
		self.embedding = PatchEmbedding(self.latent_dim, self.patch_size, self.img_height, self.img_width, name='Embedding')
		
		self.num_patches_root = int(img_height / self.patch_size)
		
		self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
		
	def call(self, inputs, training=False):
		batch_size = inputs.shape[0]
		
		patches = inputs
		embedded_patches = self.embedding(patches)
		
		if self.num_register is not None:
			register_tokens = tf.random.normal(shape=[batch_size, self.num_register, self.latent_dim])
			embedded_patches = tf.concat(embedded_patches, register_tokens)
			encoded_patches = self.encoder(embedded_patches)
			encoded_patches_wo_registers = encoded_patches[:, :-self.num_register, :]
		else:
			encoded_patches_wo_registers = self.encoder(embedded_patches)
		
		representation = keras.layers.LayerNormalization()(encoded_patches_wo_registers)
		
		representation = keras.layers.Reshape(
			target_shape=(self.num_patches_root, self.num_patches_root, self.latent_dim)
		)(representation)
		
		reconstructed = self.decoder(representation)
		
		return reconstructed
	
	@property
	def metrics(self):
		return [
			self.reconstruction_loss_tracker
		]

	def train_step(self, images):
		with tf.GradientTape() as tape:
			reconstruction = self(images, training=True)
			
			if reconstruction.shape[1:] != images.shape[1:]:
				raise ValueError('Decoder is wrong. It does not reproduce the correct shape.')
			reconstruction_loss = loss_vision(images, reconstruction)
		grads = tape.gradient(reconstruction_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		return {
				'reconstruction_loss': self.reconstruction_loss_tracker.result()
			}