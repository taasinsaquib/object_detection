# Inspired by Aladdin Persson (https://www.youtube.com/watch?v=XXYG5ZWtjj0&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=2)
# Slightly cleaned up by me :)

import torch

def iou(boxes_pred, boxes_labels, box_format=""):
	"""
	Calculates intersection over union for pairs of bounding boxes

	Params:
		boxes_pred,   Tensor, shape (N, 4)
		boxes_labels, Tensor, shape (N, 4)
			* N is the number of bboxes, each of which is represented by 4 numbers
		box_format,   String, '' or 'midpoint'
			* '', default, top left and bottom right corners (x1, y1, x2, y2)
				* used by VOC
			* 'midpoint' is center of box then width and height (x, y, w, h)
	Returns:
		Tensor, IoU for each row (N pairs of boxes)
	"""

	# Get pred and label box coordinates **************************************

	# we need the corner format to calculate intersection
	if box_format == 'midpoint':
		pred_x1 = boxes_pred[..., 0] - boxes_pred[..., 2] / 2
		pred_y1 = boxes_pred[..., 1] - boxes_pred[..., 3] / 2
		pred_x2 = boxes_pred[..., 0] + boxes_pred[..., 2] / 2
		pred_y2 = boxes_pred[..., 1] + boxes_pred[..., 3] / 2

		labels_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
		labels_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
		labels_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
		labels_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

	else:
		pred_x1 = boxes_pred[..., 0]
		pred_y1 = boxes_pred[..., 1]
		pred_x2 = boxes_pred[..., 2]
		pred_y2 = boxes_pred[..., 3]

		labels_x1 = boxes_labels[..., 0]
		labels_y1 = boxes_labels[..., 1]
		labels_x2 = boxes_labels[..., 2]
		labels_y2 = boxes_labels[..., 3]

	# Intersection ************************************************************

	# Get intersection points of each pair of boxes
	x1 = torch.max(pred_x1, labels_x1)
	y1 = torch.max(pred_y1, labels_y1)

	x2 = torch.min(pred_x2, labels_x2)
	y2 = torch.min(pred_y2, labels_y2)

	# If boxes don't intersect, either (x2 - x1) or (y2 - y1) will be negative
	# we just set the minimum to zero as the area would be zero
	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

	# Union *******************************************************************

	# Areas of prediction and label bboxes
	area_pred   = (pred_x2 - pred_x1)     * (pred_y2 - pred_y1)
	area_labels = (labels_x2 - labels_x1) * (labels_y2 - labels_y1)

	union = area_pred + area_labels - intersection

	# IoU *********************************************************************

	# add small number for numerical stability?
	return intersection / (union + 1e-6) 


def main():

	# using ... instead of : would allow one row tensors to not need to be surrounded by braces

	# # no intersection
	# pred   = torch.Tensor([[1, 1, 2, 2]])
	# labels = torch.Tensor([[3, 3, 4, 4]])
	# i = iou(pred, labels)
	# print(i)

	# # intersection
	# pred   = torch.Tensor([[1, 1, 4, 4]])
	# labels = torch.Tensor([[0, 0, 3, 3]])
	# i = iou(pred, labels)
	# print(i)

	# pred   = torch.Tensor([[0.8, 0.1, 0.2, 0.2]])
	# labels = torch.Tensor([[0.9, 0.2, 0.2, 0.2]])
	# i = iou(pred, labels, 'midpoint')
	# assert(i - 1/7 < 1e-3)

	# caught bug in midpoint
	pred   = torch.Tensor([[0.5, 0.45, 0.4, 0.5]])
	labels = torch.Tensor([[0.5, 0.5, 0.2, 0.4]])
	i = iou(pred, labels, 'midpoint')
	# assert(i - 3/13 < 1e-3)
	print(i)

if __name__ == "__main__":
	main()
