import keras_core as keras
import tensorflow as tf
import numpy as np


class PatchEmbedding(keras.layers.Layer):
	r"""

	Args:
		latent_dim: int
			The dimensions of the latent space.
		patch_size: int or tuple
			The size of the patches.
		img_height: int
			The height of the input images.
		img_width: int
			The width of the input images.
	"""
	def __init__(
			self,
			latent_dim: int,
			patch_size: int,
			img_height: int,
			img_width: int,
			name: str | None = None
	):
		super().__init__(name=name)
		self.latent_dim = latent_dim

		if not isinstance(patch_size, (list, tuple)):
			self.patch_size = (patch_size, patch_size)
		else:
			self.patch_size = patch_size

		if (img_height % self.patch_size[0] != 0) or (img_width % self.patch_size[1] != 0):
			raise ValueError("The patch size is not compatible with the image height or image width.")

		self.img_height = img_height
		self.img_width = img_width
		self.num_patches = int((img_height / self.patch_size[0]) * (img_width / self.patch_size[1]))

		self.projection = [
			keras.layers.Conv2D(filters=self.latent_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='valid'),
			keras.layers.Reshape(target_shape=(self.num_patches, self.latent_dim))
		]

		self.positional_encoding = keras.layers.Embedding(input_dim=self.num_patches, output_dim=latent_dim)

	def call(self, patch):
		positions = tf.range(start=0, limit=self.num_patches, delta=1)
		
		for layer in self.projection:
			patch = layer(patch)
			
		encoded = patch + self.positional_encoding(positions)

		return encoded


class VisionTransformerMLP(keras.layers.Layer):
	def __init__(
			self,
			hidden_units: int,
			num_outputs: int,
			dropout: float = 0.5
	):
		super().__init__()
		self.hidden_units = hidden_units
		self.num_outputs = num_outputs
		self.dropout = dropout

		self.layers = [
			keras.layers.Dense(units=2 * self.hidden_units, activation='gelu'),
			keras.layers.Dropout(rate=self.dropout),
			keras.layers.Dense(units=self.num_outputs, activation='gelu'),
			keras.layers.Dropout(rate=self.dropout)
		]
		
	def call(self, inputs):
		x = inputs

		for layer in self.layers:
			x = layer(x)

		return x


class VisionTransformerEncoder(keras.layers.Layer):
	"""

	Args:
		num_heads:
		key_dim:
		hidden_units:
		num_outputs:
		name:
	"""
	def __init__(
			self,
			num_heads: int,
			key_dim: int,
			hidden_units: int,
			num_outputs: int,
			name: str | None = None
	):
		super().__init__(name=name)
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.hidden_units = hidden_units
		self.num_outputs = num_outputs
		self.AttentionLayerNormalization = keras.layers.LayerNormalization()
		self.MultiHeadAttention = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
		self.AttentionLayerNormalization = keras.layers.LayerNormalization()
		self.MLP = VisionTransformerMLP(hidden_units=self.hidden_units, num_outputs=self.num_outputs)
	
	def call(self, inputs):
		x1 = inputs
		x2 = self.AttentionLayerNormalization(x1)
		attention_output = self.MultiHeadAttention(x2, x2)
		y1 = keras.layers.Add()([attention_output, x1])
		y2 = self.AttentionLayerNormalization(y1)
		y2 = self.MLP(y2)
		encoded_inputs = keras.layers.Add()([y1, y2])
		
		return encoded_inputs
	
def get_decoder(activation='gelu', kernel_initializer='glorot_uniform'):
	"""
	
	Args:
		activation:

	Returns:

	"""
	decoder = keras.Sequential(name='Decoder')
	
	for i in range(4):
		decoder.add(keras.layers.Conv2DTranspose(
			filters=int(2**(3-i)+2),
			kernel_size=2,
			strides=2,
			padding='valid',
			activation=activation,
			kernel_initializer=kernel_initializer
		))
		decoder.add(keras.layers.BatchNormalization())
		
	return decoder
	
	
