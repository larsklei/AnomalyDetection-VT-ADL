import tensorflow as tf
import keras_core as keras
import numpy as np
import matplotlib.pyplot as plt

from scoring.custom_metrics import iou_score


def compute_grey_masks(original_images, recreated_images, threshold=0.01):
	r"""Computes a grey mask which showcases difference between original images
	and their recreated counterparts.

	Args:
		original_images: A sequence of images.
		recreated_images: The recreated image by the model
		threshold: A number used to set the difference threshold
			between the original images and recreated images

	Returns:
		mask: An array with 0, 1 entries where 1 at position (i,j) denotes that difference between original image
			and recreated image is larger than threshold.
	"""
	mean_squared = keras.metrics.MSE(original_images, recreated_images).numpy()
	mask = (mean_squared > threshold).astype(int)

	return mask


def create_sample_grid_image(images, ground_truth, predicted_masks):
	r"""

	Args:
		images:
		ground_truth:
		predicted_masks:

	Returns:
		fig:
	"""
	batch_size = images.shape[0]
	sample_size = 3

	chosen_images_index = np.random.randint(high=batch_size, size=sample_size)

	fig, axs = plt.subplots(nrows=3, ncols=3)

	row_pos = 0

	for index in chosen_images_index:
		axs[row_pos, 0].axis('off')
		axs[row_pos, 0].imshow(images[index])
		axs[row_pos, 0].set(xlabel='Original image')

		axs[row_pos, 1].axis('off')
		axs[row_pos, 1].imshow(images[index])
		axs[row_pos, 1].imshow(ground_truth[index], 'jet', interpolation='none', alpha=0.5)
		axs[row_pos, 1].set(xlabel='Ground Truth')

		axs[row_pos, 2].axis('off')
		axs[row_pos, 2].imshow(images[index])
		axs[row_pos, 2].imshow(predicted_masks[index], 'jet', interpolation='none', alpha=0.5)
		axs[row_pos, 2].set(xlabel='Predicted mask')

		row_pos += 1

	return fig


def evaluation(model, images, ground_truth, threshold=0.01):
	#recreated_images = model.predict(images)
	#predicted_masks = compute_grey_masks(images, recreated_images, threshold)

	#grid_sample_figure = create_sample_grid_image(images, ground_truth, predicted_masks)
	#grid_sample_figure.show()
	#grid_sample_figure.savefig('sample_evaluation.png')

	#score_predicted_masks = 1 - tf.reduce_mean(tf.image.ssim(ground_truth, predicted_masks))
	#iou_scores = iou_score(ground_truth, predicted_masks)
	pass