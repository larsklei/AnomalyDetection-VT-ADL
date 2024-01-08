import keras_core as keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
class PatchEmbedding(keras.layers.Layer):
	"""A simple implementation of the patch embedding using an inner convolution layer as Keras layer object.

	Args:
		embed_dim: int
			The dimension of the latent space.
		patch_size: int | tuple | list
			A integer or a tuple of integers to be the size of the patches
		img_height: int
			The height of the image file
		img_width: int
			The width of the image file
	
	Attributes:
		name: The name of the layer
	"""
	def __init__(
			self,
			embed_dim: int,
			patch_size: int,
			img_height: int,
			img_width: int,
			name: str | None = None,
			**kwargs
	):
		super().__init__(name=name, **kwargs)
		self.embed_dim = embed_dim

		if not isinstance(patch_size, (list, tuple)):
			self.patch_size = (patch_size, patch_size)
		else:
			self.patch_size = patch_size

		if (img_height % self.patch_size[0] != 0) or (img_width % self.patch_size[1] != 0):
			raise ValueError("The patch size is not compatible with the image height or image width.")

		self.img_height = img_height
		self.img_width = img_width
		self.num_patches = int((img_height / self.patch_size[0]) * (img_width / self.patch_size[1]))

		self.embedding = [
			keras.layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='valid'),
			keras.layers.Reshape(target_shape=(self.num_patches, self.embed_dim))
		]

		self.positional_encoding = keras.layers.Embedding(input_dim=self.num_patches, output_dim=embed_dim)

	def call(self, patch):
		positions = tf.range(start=0, limit=self.num_patches, delta=1)
		
		for layer in self.embedding:
			patch = layer(patch)
			
		encoded = patch + self.positional_encoding(positions)

		return encoded
	
	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"proj_dim": self.embed_dim,
				"patch_size": self.patch_size,
				"img_height": self.img_height,
				"img_width": self.img_width
			}
		)
		return config

@keras.saving.register_keras_serializable()
class VisionTransformerBlock(keras.layers.Layer):
	"""A simple implementation of the standard VisionTransformer Encoder Block
	architecture.

	Args:
		embed_dim: int
			The key dimension in MultiHeadAttention
		num_heads: int
			The number of heads used in MultiHeadAttention
		hidden_unit: int
			The number of hidden units in
		num_outputs: int
			The output of the block.
		
	Attributes:
		rate: float
			The rate of the dropout layer in the MLP block.
		name: str
			The name of the block.
	"""
	
	def __init__(
			self,
			embed_dim: int,
			num_heads: int,
			hidden_unit: int,
			rate: float = 0.2,
			name: str | None = None,
			**kwargs
	):
		super().__init__(name=name, **kwargs)
		self.num_heads = num_heads
		self.embed_dim = embed_dim
		self.hidden_unit = hidden_unit
		self.attention = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
		self.rate = rate
		self.mlp = keras.Sequential([
			keras.layers.Dense(self.hidden_unit, activation="gelu"),
			keras.layers.Dropout(rate),
			keras.layers.Dense(self.embed_dim)
		])
		self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout = keras.layers.Dropout(rate)
	
	def call(self, inputs):
		norm1_output = self.layernorm1(inputs)
		att_output = self.attention(norm1_output, norm1_output)
		dropout_output = self.dropout(att_output)
		out1 = keras.layers.Add()([inputs, dropout_output])
		
		norm2_output = self.layernorm2(out1)
		mlp_output = self.mlp(norm2_output)
		encoded_inputs = keras.layers.Add()([out1, mlp_output])
		return encoded_inputs
	
	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"num_heads": self.num_heads,
				"embed_dim": self.embed_dim,
				"hidden_unit": self.hidden_unit,
				"rate": self.rate
			}
		)
		
		return config
	
def get_decoder(activation='gelu', kernel_initializer='glorot_uniform'):
	"""
	
	Attributes:
		activation:
		kernel_initializer:
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
