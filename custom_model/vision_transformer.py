import keras_core as keras
import numpy as np

class PatchEmbedding(keras.layers.Layer):
	r"""

	Args:
		latent_dim: The dimensions of the latent space.
		patch_size: The size of the patches.
		img_height: The height of the input images.
		img_width: The width of the input images.
	"""

	def __init__(self, latent_dim, patch_size, img_height, img_width, name):
		super().__init__(trainable=False,
						 name=name)

		self.latent_dim = latent_dim

		if not isinstance(patch_size, (list, tuple)):
			self.patch_size = (patch_size, patch_size)
		else:
			self.patch_size = patch_size

		if not (img_height/self.patch_size[0] % 1 == 0) && (img_width/self.patch_size[1] % 1 == 0):
			raise ValueError("The patch size is not compatible with the image height or image width.")

		self.img_height = img_height
		self.img_width = img_width
		self.num_patches = (img_height/self.patch_size[0])*(img_width/self.patch_size[1])

		self.projection = keras.layers.Conv2D(filters=self.latent_dim,
								kernel_size=self.patch_size,
								strides=self.patch_size,
								padding='valid')

		self.positional_encoding = keras.layers.Embedding(input_dim=self.num_patches, output_dim=self.latent_dim)

	def call(self, inputs):
		positions = np.arange(start=0, stop=self.num_patches)
		encoded = self.projection(inputs)+self.positional_encoding(positions)

		return encoded

class VisionTransformer(keras.layers.Layer):
	raise NotImplemented