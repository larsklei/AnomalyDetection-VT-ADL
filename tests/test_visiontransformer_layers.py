import tensorflow as tf

from custom_model.visiontransformer_layers import PatchEmbedding, VisionTransformerMLP, VisionTransformerEncoder


class PatchEmbeddingTest(tf.test.TestCase):
	def test_PatchEmbeddingOutput(self):
		batch_size = 4
		latent_dim = 32
		patch_size = 16
		img_height = 32
		img_width = 32
		channels = 3
		input_shape = (batch_size, img_height, img_width, channels)
		dummy_images = tf.ones(input_shape)
		
		num_patches = 4
		expected_output_shape = (batch_size, num_patches, latent_dim)
		
		emb = PatchEmbedding(
			latent_dim=latent_dim,
			patch_size=patch_size,
			img_height=img_height,
			img_width=img_width
		)
		
		output = emb(dummy_images)
		
		self.assertEqual(output.shape, expected_output_shape)
	

class VisionTransformerMLPTest(tf.test.TestCase):
	def test_ViTMLPOutput(self):
		batch_size = 4
		hidden_units = 16
		num_outputs = 16
		dim = 4
		
		input_shape = (batch_size, dim)
		dummy_data = tf.ones(input_shape)
		
		expected_output_shape = (batch_size, num_outputs)
		
		mlp = VisionTransformerMLP(hidden_units=hidden_units, num_outputs=num_outputs)
		
		output = mlp(dummy_data)
		
		self.assertEqual(output.shape, expected_output_shape)
		

if __name__ == '__main__':
	tf.test.main()