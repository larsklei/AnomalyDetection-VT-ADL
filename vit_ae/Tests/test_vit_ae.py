import keras

from keras.src import testing
from absl.testing import parameterized

from custom_model.visiontransformer_autoencoder import VisionTransformerAutoencoder

class TestVisionTransformerAutoEncoder(testing.TestCase, parameterized.TestCase):
	
	def test_ViTAutoEncoder_serialization_return(self):
		encoder = keras.Sequential([
			keras.layers.Activation(activation=keras.activations.linear)
		])
		decoder = keras.Sequential([
			keras.layers.Dense(units=768),
			keras.layers.Reshape((16, 16, 3))
		])
		
		model = VisionTransformerAutoencoder(
			embed_dim=4,
			patch_size=16,
			encoder=encoder,
			decoder=decoder,
			img_height=16,
			img_width=16
		)
		self.assertIsInstance(model, keras.Model)
		
		revived = self.run_class_serialization_test(model)
		
		self.assertEqual(len(revived._layers), 5)
		
	def test_ViTAutoEncoder_OutputError(self):
		encoder = keras.Sequential([
			keras.layers.Activation(activation=keras.activations.linear)
		])
		decoder = keras.Sequential([
			keras.layers.Dense(units=256),
			keras.layers.Reshape((16, 16, 1))
		])
		
		model = VisionTransformerAutoencoder(
			embed_dim=4,
			patch_size=16,
			encoder=encoder,
			decoder=decoder,
			img_height=16,
			img_width=16
		)
		
		input_shape = (4, 16, 16, 3)
		dummy_images = keras.ops.ones(input_shape)
		
		self.assertRaises(ValueError, model.fit, dummy_images)
		
		
if __name__ == "__main__":
	testing.main()