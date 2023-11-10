import tensorflow as tf
import keras_core as keras

from custom_model.visiontransformer_layers import PatchEmbedding, VisionTransformerEncoder, get_decoder
from common.model_utilities.loss_metric_utilities import VisionLoss

class VisionTransformerAutoencoder(keras.Model):
	"""

	Args:
		latent_dim: int
		
		patch_size: int
		
		encoder: keras.Model
		
		decoder: keras.Model
		
		img_height: int
		
		img_width: int
		
		num_register: int
	"""
	def __init__(
			self,
			latent_dim: int,
			patch_size: int,
			encoder: keras.Model,
			decoder: keras.Model,
			img_height: int,
			img_width: int,
			num_register: int | None = None,
			latent_layers: int | None = None,
			name: str | None = None
	):
		if (img_height % patch_size != 0) or (img_width % patch_size != 0):
			raise ValueError("Image height or width is not divisible by patch_size.")
		super().__init__(name=name)
		self.latent_dim = latent_dim
		self.patch_size = patch_size
		self.encoder = encoder
		self.decoder = decoder
		self.img_height = img_height
		self.img_width = img_width
		self.num_register = num_register
		self.num_patches_root = tf.cast(img_height / self.patch_size, "int32")
		self.embedding = PatchEmbedding(
			self.latent_dim,
			self.patch_size,
			self.img_height,
			self.img_width,
			name="Embedding"
		)
	
		self.latent_layers = [
			keras.layers.LayerNormalization(),
			keras.layers.Reshape(
				target_shape=(self.num_patches_root, self.num_patches_root, self.latent_dim)
			)
		]

		self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
		
	def call(self, inputs, training=False):
		batch_size = inputs.shape[0]
		
		patches = inputs
		embedded_patches = self.embedding(patches)
		
		if self.num_register is not None:
			register_tokens = tf.random.normal(shape=(batch_size, self.num_register, self.latent_dim))
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
			self.reconstruction_loss_tracker
		]

	def train_step(self, images):
		with tf.GradientTape() as tape:
			reconstruction = self(images, training=True)
			
			if reconstruction.shape[1:] != images.shape[1:]:
				raise ValueError("Decoder is wrong. It does not reproduce the correct shape.")
			reconstruction_loss = VisionLoss()(images, reconstruction)
		grads = tape.gradient(reconstruction_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		return {
			"reconstruction_loss": self.reconstruction_loss_tracker.result()
		}
		
	
if __name__ == "__main__":
	decoder = get_decoder()
	encoder = keras.Sequential(
		[
			VisionTransformerEncoder(num_heads=4, key_dim=512, hidden_units=32, num_outputs=512),
			VisionTransformerEncoder(num_heads=4, key_dim=512, hidden_units=32, num_outputs=512)
		]
	)

	input_shape = (4, 512, 512, 3)

	ViT = VisionTransformerAutoencoder(
		latent_dim=512,
		patch_size=16,
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

	ViT.fit(input_dummy)