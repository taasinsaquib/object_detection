# Inspired by Aladdin Persson: https://www.youtube.com/watch?v=YDkjWEN8jNA&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=3

import torch
import matplotlib.pyplot as plt

from iou import iou
from utils_draw import draw_bboxes


def visualize_nms(boxes, nms_boxes, num_classes, box_format=''):
	"""
	Draw bboxes with matplotlib

	Params:
	boxes, Tensor, shape (N, 6)
		* N is the number of bboxes, each of which is represented by 6 numbers
		* [class, confidence, x1, y1, x2, y2] or midpoint format for the last 4 numbers
	box_format,   String, '' or 'midpoint'
		* '', default, top left and bottom right corners (x1, y1, x2, y2)
			* used by VOC
		* 'midpoint' is center of box then width and height (x, y, w, h)
	"""

	fig, ax = plt.subplots(1, 2)

	draw_bboxes(ax[0], "Detections", boxes, box_format)
	draw_bboxes(ax[1], "NMS'd Detections", nms_boxes, box_format)

	plt.show()


def nms(pred, iou_threshold, prob_threshold, box_format=''):
	"""
	Perform non-max supression for multiple bbox guesses

	Params:
		pred,   Tensor, shape (N, 6)
			* N is the number of bboxes, each of which is represented by 6 numbers
			* [class, confidence, x1, y1, x2, y2] or midpoint format for the last 4 numbers
		iou_threshold, Float, threshold to remove overlapping bboxes
		prob_threshold, Float, threshold to remove low-confidence bboxes
		box_format,   String, '' or 'midpoint'
			* '', default, top left and bottom right corners (x1, y1, x2, y2)
				* used by VOC
			* 'midpoint' is center of box then width and height (x, y, w, h)
	Returns:
		Tensor, IoU for each row (N pairs of boxes)
	"""

	# print(pred)

	bboxes = [box for box in pred if box[1] > prob_threshold]	# remove low confidence bboxes
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)	# sort in descending order of confidence

	bboxes_nms = []

	while bboxes:
		chosen_box = bboxes.pop(0)

		# keep boxes of a different class from or low IoU with the chosen box
		bboxes = [
			box for box in bboxes
			if box[0] != chosen_box[0]
			or iou(torch.Tensor([chosen_box[2:]]), torch.Tensor([box[2:]]), box_format=box_format) < iou_threshold
		]

		bboxes_nms.append(chosen_box)

	return bboxes_nms


def main():

	# Test 1
	boxes = [
		[1, 1, 0.5, 0.45, 0.4, 0.5],
		[1, 0.8, 0.5, 0.5, 0.2, 0.4],
		[1, 0.7, 0.25, 0.35, 0.3, 0.1],
		[1, 0.05, 0.1, 0.1, 0.1, 0.1],
	]
	answer = [[1, 1, 0.5, 0.45, 0.4, 0.5], [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	# visualize_nms(boxes, nms_boxes, 1, 'midpoint')
	assert(sorted(nms_boxes) == sorted(answer))

	# Test 2
	boxes = [
		[1, 1, 0.5, 0.45, 0.4, 0.5],
		[2, 0.9, 0.5, 0.5, 0.2, 0.4],
		[1, 0.8, 0.25, 0.35, 0.3, 0.1],
		[1, 0.05, 0.1, 0.1, 0.1, 0.1],
	]
	answer = [[1, 1, 0.5, 0.45, 0.4, 0.5], [2, 0.9, 0.5, 0.5, 0.2, 0.4], [1, 0.8, 0.25, 0.35, 0.3, 0.1],]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	# visualize_nms(boxes, nms_boxes, 2, 'midpoint')
	assert(sorted(nms_boxes) == sorted(answer))

	# Test 3
	boxes = [
		[1, 0.9, 0.5, 0.45, 0.4, 0.5],
		[1, 1, 0.5, 0.5, 0.2, 0.4],
		[2, 0.8, 0.25, 0.35, 0.3, 0.1],
		[1, 0.05, 0.1, 0.1, 0.1, 0.1],
	]
	answer = [[1, 1, 0.5, 0.5, 0.2, 0.4], [2, 0.8, 0.25, 0.35, 0.3, 0.1]]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	# visualize_nms(boxes, nms_boxes, 2, 'midpoint')
	assert(sorted(nms_boxes) == sorted(answer))

	# Test 4
	boxes = [
		[1, 0.9, 0.5, 0.45, 0.4, 0.5],
		[1, 1, 0.5, 0.5, 0.2, 0.4],
		[1, 0.8, 0.25, 0.35, 0.3, 0.1],
		[1, 0.05, 0.1, 0.1, 0.1, 0.1],
	]
	answer = [[1, 0.9, 0.5, 0.45, 0.4, 0.5], [1, 1, 0.5, 0.5, 0.2, 0.4], [1, 0.8, 0.25, 0.35, 0.3, 0.1]]

	nms_boxes = nms(boxes, 9/20, 0.2, 'midpoint')
	visualize_nms(boxes, nms_boxes, 2, 'midpoint')
	assert(sorted(nms_boxes) == sorted(answer))


if __name__ == "__main__":
	main()
