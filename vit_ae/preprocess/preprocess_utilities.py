import tensorflow as tf
import keras
import random

from pathlib import Path
from sklearn.model_selection import train_test_split

OBJECT_NAMES = {
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

IMG_HEIGHT = 512
IMG_WIDTH = 512
CHANNELS = 3


def load_image(image_file):
	image = keras.utils.load_img(image_file)
	image_arr = keras.utils.img_to_array(image)
	image_arr = keras.ops.cast(image_arr, dtype="float32")
	
	return image_arr


def resize(image, height=IMG_HEIGHT, width=IMG_WIDTH):
	return keras.ops.image.resize(image, [height, width], interpolation="nearest")


def normalize(image):
	return image / 255.0


def image_preprocess(image_file):
	image = load_image(image_file)
	image = resize(image)
	image = normalize(image)
	return image


def get_ground_truth_image(image_file, height=IMG_HEIGHT, width=IMG_WIDTH, channels=CHANNELS):
	"""Returns the ground image associated to the image file. If image file has defect "good" or is part of the
	training set, a zero tensor of the shape (height, width, channels) will be returned.
	
	Args:
		image_file:
			Path to the image file
		height: int
			Height of the image
		width: int
			Width of the image
		channels: int
			Number of channels of the image.

	Returns:
		gt_image: Array of floats.

	"""
	stem = image_file.stem
	path_parts = list(image_file.parts)
	if path_parts[-2] == "good":
		gt_image = tf.zeros((height, width, channels))
	else:
		path_parts[-1] = stem + "_mask.png"
		path_parts[-3] = "ground_truth"
		gt_path = Path(path_parts[0], *path_parts[1:])
		gt_image = image_preprocess(gt_path)
	
	return gt_image


def get_image_files(
		source_path,
		data_objects=None,
		training=True
):
	"""Returns the paths to the images of the objects in data objects.
	
	Args:
		source_path:
			Path to the MVTEC AD dataset directory.
		data_objects: str | list of strings | None
			A subset of the objects of the MVTEC AD dataset
		training: boolean
			A value to see if the training dataset is generated

	Returns:
		A list of image file paths.
	"""
	if data_objects is None:
		data_objects = list(OBJECT_NAMES)
	if isinstance(data_objects, str):
		data_objects = [data_objects]
	if not set(data_objects).issubset(OBJECT_NAMES):
		raise ValueError("The argument data_objects contains strings which are not part of the MVTEC-AD objects.")
	source_path = Path(source_path)
	if training:
		stem_paths = [f"{data_object}/train/*/*.png" for data_object in data_objects]
	else:
		stem_paths = [f"{data_object}/test/*/*.png" for data_object in data_objects]
	image_files = []
	for stem_path in stem_paths:
		image_files += list(source_path.glob(stem_path))
	
	return image_files


class DataGenerator:
	"""Returns a set of images with their ground truth images.

	Args:
		image_files:
			Image file paths.
		training: bool
			Whether to create the training dataset.
			
	Returns:
		A data generator which returns the images.
	"""
	
	def __init__(self, image_files, training: bool = False):
		self.image_files = image_files
		self.training = training
	
	def __call__(self):
		if self.training:
			random.shuffle(self.image_files)
		
		for image_file in self.image_files:
			image = image_preprocess(image_file)
			ground_truth = get_ground_truth_image(image_file)
			
			yield image, ground_truth


def get_train_val_split(image_files, validation_split=0.2):
	"""Creates training and validation split for the image files.
	
	Args:
		image_files:
			Image file paths.
		validation_split: float
			The ratio of the validation split

	Returns:
		Two arrays of paths to the training and validation dataset.
	"""
	train_files, val_files = train_test_split(image_files, test_size=validation_split)
	
	return train_files, val_files
