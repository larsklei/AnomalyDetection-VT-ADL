import keras_core as keras

from pathlib import Path


def get_train_data(data_object, dataset_dir):
	"""
	
	Args:
		data_object: str
			Name of object which
		dataset_dir:

	Returns:
		ds_train: Tensorflow Dataset
	"""
	path_dataset_dir = Path(dataset_dir)
	path_train_dataset = path_dataset_dir.joinpath(data_object, 'train')
	ds_train = keras.utils.image_dataset_from_directory(
		directory=path_train_dataset,
		labels=None,
		image_size=(512, 512)
	)
	return ds_train

def get_eval_data(data_object, dataset_dir):
	"""
	
	Args:
		data_object: str
		
		dataset_dir: Directory where the data is located.

	Returns:

	"""
	path_dataset_dir = Path(dataset_dir)
	path_test_dataset = path_dataset_dir.joinpath(data_object, 'test')
	path_gt_dataset = path_dataset_dir.joinpath(data_object, 'ground_truth')
	ds_test_dataset = keras.utils.image_dataset_from_directory(
		directory=path_test_dataset,
		labels='inferred',
		image_size=(512, 512)
	)
	ds_gt_dataset = keras.utils.image_dataset_from_directory(
		directory=path_gt_dataset,
		labels='inferred',
		image_size=(512, 512)
	)
	return ds_test_dataset, ds_gt_dataset
