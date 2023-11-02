import numpy as np
from tensorflow import keras


def iou_score(y_mask_pred, y_mask_true):
	r"""
	Computes the Intersection over Union score for a prediction mask and its ground truth

	Args:
		y_mask_pred (): The mask of the predicted image.
		y_mask_true (): The ground truth.

	Returns:
		iou_score (float): The intersection over union score.
	"""
	if y_mask_pred.shape == y_mask_true.shape:
		intersection = y_mask_pred * y_mask_true
		union = y_mask_pred + y_mask_true - intersection

		count_intersection = np.sum(intersection)
		count_union = np.sum(union)

		iou_score = count_intersection / count_union

		return iou_score
	else:
		raise ValueError("Shapes of the input does not match.")