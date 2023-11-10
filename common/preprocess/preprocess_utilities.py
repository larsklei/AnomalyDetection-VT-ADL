import tensorflow as tf
import keras_core as keras

from pathlib import Path
from typing import Mapping, Any, Union, Optional

NAME_OBJECTS = {
	'bottle',
	'cable',
	'capsule',
	'carpet',
	'grid',
	'hazelnut',
	'leather',
	'metal_nut',
	'pill',
	'screw',
	'tile',
	'toothbrush',
	'transistor',
	'wood',
	'zipper'
}

def load_image(image_file):
	image = tf.io.read_file(image_file)
	image = tf.io.decode_png(image)
	
	image = tf.cast(image, tf.float32)
	
	return image

def resize(image, height=512, width=512):
	return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def crop_image(image):
	pass

def normalize(image):
	return image / 255.0

def image_preprocess(image_file):
	image = load_image(image_file)
	image = resize(image)
	image = crop_image(image)
	image = normalize(image)
	
	return image

def get_ground_truth(image_file: Path, height=512, width=512):
	parts = list(file_path.parts)
	if parts[-2] == "good":
		ground_truth = tf.zeros(shape=(height, width, 3))
	else:
		parts[-3] = "ground_truth"
		ground_truth_path = Path(parts[0],*parts[1:])
		ground_truth = image_preprocess(ground_truth_path)
	
	return ground_truth

def eval_data_pipeline(image_file):
	original_image = image_preprocess(image_file)
	ground_truth = get_ground_truth(image_file)
	
	return original_image, ground_truth
		

if __name__ == "__main__":
	path = Path("/Users/larskleinemeier/Documents/AnomalyDetection-VT-ADL/data_images/bottle/test/broken_large/000.png")
	
	parts = list(path.parts)
	parts[-3] = "ground_truth"
	
	path2 = Path(parts[0], *parts[1:])
	print(str(path))
	print(str(path2))
	
	
	