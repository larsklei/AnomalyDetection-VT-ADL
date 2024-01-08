import keras
import tensorflow as tf

from keras.src import testing
from absl.testing import parameterized
from vit_ae.custom_model.visiontransformer_layers import PatchEmbedding, VisionTransformerBlock


class PatchEmbeddingTest(testing.TestCase, parameterized.TestCase):
	
	def test_PatchEmbedding_serialization(self):
		layer = PatchEmbedding(128, 16, 256, 256)

		revived = self.run_class_serialization_test(layer)

		self.assertLen(revived._layers, 3)
	
	@parameterized.named_parameters(
		("correct_output_1", 4, 28, 16, 256, 256, 3),
		("correct_output_2", 16, 28, 32, 128, 128, 3)
	)
	def test_PatchEmbeddingOutput(self, batch_size, latent_dim, patch_size, img_height, img_width, channels):
		input_shape = (batch_size, img_height, img_width, channels)
		dummy_images = keras.ops.ones(input_shape)
		
		num_patches = int((img_height * img_width) / (patch_size * patch_size))
		
		expected_output_shape = (batch_size, num_patches, latent_dim)
		
		emb = PatchEmbedding(
			embed_dim=latent_dim,
			patch_size=patch_size,
			img_height=img_height,
			img_width=img_width
		)
		
		output = emb(dummy_images)
		
		self.assertEqual(output.shape, expected_output_shape)
	
	def test_PatchEmbedding_Error(self):
		self.assertRaises(ValueError, PatchEmbedding, 8, 5, 256, 256)


class VisionTransformerBlockTest(testing.TestCase, parameterized.TestCase):
	def test_ViTBlock_serialization(self):
		layer = VisionTransformerBlock(4, 1, 16, rate=0.3, name="block")

		revived = self.run_class_serialization_test(layer)

		self.assertLen(revived._layers, 5)

	@parameterized.named_parameters(
		("correct_output_1", 64, 2, 32, 2, 128),
		("correct_output_2", 4, 2, 128, 2, 3)
	)
	def test_ViTBlockOutput(self, batch_size, num_patches, embed_dim, num_heads, hidden_unit):
		layer = VisionTransformerBlock(embed_dim, num_heads, hidden_unit)
		input_shape = (batch_size, num_patches, embed_dim)
		dummy_data = tf.random.normal(input_shape)

		output = layer(dummy_data)

		self.assertEqual(output.shape, input_shape)


if __name__ == "__main__":
	testing.main()