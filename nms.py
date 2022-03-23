# Inspired by Aladdin Persson: https://www.youtube.com/watch?v=YDkjWEN8jNA&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=3

import torch
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

from iou import iou

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)

def visualize_bboxes(boxes, nms_boxes, num_classes, box_format=''):
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

	# Display the image
	img = torch.full((2, 2, 3), 255, dtype=torch.uint8)
	ax[0].imshow(img)
	ax[1].imshow(img)

	# cmap = get_cmap(num_classes)
	cmap = {1: 'r', 2: 'g', 3: 'b'}

	# Create rectangle patches
	for box in boxes:
		if box_format == 'midpoint':
			rect = Rectangle((box[2] - box[4]/2, box[3] - box[5]/2), box[4], box[5], linewidth=box[1], edgecolor=cmap[box[0]], facecolor='none')
		else:
			rect = Rectangle((box[2], box[3]), box[4]-box[2], box[5]-box[3], linewidth=box[1], edgecolor=cmap[box[0]], facecolor='none')

		ax[0].add_patch(rect)

	# Create rectangle patches
	for box in nms_boxes:
		if box_format == 'midpoint':
			rect = Rectangle((box[2] - box[4]/2, box[3] - box[5]/2), box[4], box[5], linewidth=box[1], edgecolor=cmap[box[0]], facecolor='none')
		else:
			rect = Rectangle((box[2], box[3]), box[4]-box[2], box[5]-box[3], linewidth=box[1], edgecolor=cmap[box[0]], facecolor='none')

		ax[1].add_patch(rect)

	ax[0].set_title("Detections")
	ax[1].set_title("NMS'd Detections")
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

	bboxes = [box for box in pred if box[1] > prob_threshold]	# remove low confidence bboxes
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)	# sort in descending order of confidence

	bboxes_nms = []

	while bboxes:
		print(bboxes)
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
	visualize_bboxes(boxes, nms_boxes, 1, 'midpoint')
	assert(sorted(nms_boxes) == sorted(answer))

	# Test 2
	boxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [2, 0.9, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	visualize_bboxes(boxes, nms_boxes, 2)

	answer = [[1, 1, 0.5, 0.45, 0.4, 0.5], [2, 0.9, 0.5, 0.5, 0.2, 0.4], [1, 0.8, 0.25, 0.35, 0.3, 0.1],]

	assert(sorted(nms_boxes) == sorted(answer))

	# Test 3
	boxes = [
        [1, 0.9, 0.5, 0.45, 0.4, 0.5],
        [1, 1, 0.5, 0.5, 0.2, 0.4],
        [2, 0.8, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	visualize_bboxes(boxes, nms_boxes, 2)

	answer = [[1, 1, 0.5, 0.5, 0.2, 0.4], [2, 0.8, 0.25, 0.35, 0.3, 0.1]]

	assert(sorted(nms_boxes) == sorted(answer))

	# Test 3
	boxes = [
        [1, 0.9, 0.5, 0.45, 0.4, 0.5],
        [1, 1, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]

	nms_boxes = nms(boxes, 7/20, 0.2, 'midpoint')
	visualize_bboxes(boxes, nms_boxes, 2)

	answer = [[1, 0.9, 0.5, 0.45, 0.4, 0.5], [1, 1, 0.5, 0.5, 0.2, 0.4], [1, 0.8, 0.25, 0.35, 0.3, 0.1]]

	assert(sorted(nms_boxes) == sorted(answer))

if __name__ == "__main__":
	main()
