# Inspired by Aladdin Persson: https://www.youtube.com/watch?v=FppOzcDvaDI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=4

import torch
from collections import Counter

from iou import iou

def map(boxes_pred, boxes_labels, iou_threshold=0.5, box_format='', num_classes=20):
	"""
	Calculate mean average precision for predicted bounding boxes

	Params:
		boxes_pred,   Tensor, shape (N, 7)
			* N is the number of bboxes, each of which is represented by 6 numbers
			* [train_idx, class_pred, confidence, x1, y1, x2, y2] or midpoint format for the last 4 numbers
		iou_threshold, Float, threshold to compare against ground truth bbox
		box_format,   String, '' or 'midpoint'
			* '', default, top left and bottom right corners (x1, y1, x2, y2)
				* used by VOC
			* 'midpoint' is center of box then width and height (x, y, w, h)
	Returns:
		Tensor, IoU for each row (N pairs of boxes)
	"""

	avg_precisions = []
	eps = 1e-6

	# map each possible class -> associated detections and ground truths
	detections_map = {}
	ground_truths_map = {}

	for p in boxes_pred:
		c = p[1]

		if c in detections_map:
			detections_map[c].append(p)
		else:
			detections_map[c] = [p]

	for l in boxes_labels:
		c = l[1]

		if c in ground_truths_map:
			ground_truths_map[c].append(l)
		else:
			ground_truths_map[c] = [l]


	# find AP for each class **************************************************
	for c in range(num_classes):

		# nothing was detected for this class, skip
		if c not in ground_truths_map:
			continue
		ground_truths = ground_truths_map[c]

		# collect detection and label bboxes for the current class
		if c not in detections_map:
			detections = []
		else:
			detections = detections_map[c]

		# map each image idx to the number of label bboxes it contains
		num_bboxes = Counter(gt[0] for gt in ground_truths)

		# going to keep track of which bbox from each image we've looked at?
		for k, v in num_bboxes.items():
			num_bboxes[k] = torch.zeros(v)

		# sort detections in descending order of confidence score
		detections.sort(key=lambda x: x[2], reverse=True)

		# keep track of TP and FP as we go, not an int because we want a cumulative sum later
		TP = torch.zeros((len(detections)))
		FP = torch.zeros((len(detections)))
		num_bboxes_labels = len(ground_truths)

		# map image index to the ground truth bboxes it contains
		ground_truth_idx_map = {}
		for bbox in ground_truths:
			idx = bbox[0]

			if idx in ground_truth_idx_map:
				ground_truth_idx_map[idx].append(bbox)
			else:
				ground_truth_idx_map[idx] = [bbox]

		# TODO: remove detections as we go
		for detection_idx, detection in enumerate(detections):

			# get possible labels for a given detected bbox
			ground_truth_img = ground_truth_idx_map[detection[0]]

			num_ground_truth = len(ground_truth_img)
			best_iou = 0
			best_gt_idx = -1

			for i, gt in enumerate(ground_truth_img):
				iou_score = iou(torch.Tensor([detection[3:]]), torch.Tensor([gt[3:]]), box_format=box_format)

				if iou_score > best_iou:
					best_iou = iou_score
					best_gt_idx = i

			if best_iou > iou_threshold:
				# we haven't assigned this detection to a label yet
				if num_bboxes[detection[0]][best_gt_idx] == 0:
					TP[detection_idx] = 1
					num_bboxes[detection[0]][best_gt_idx] = 1
				# detection already assigned to a label, so this is a FP
				else:
					FP[detection_idx] = 1
			# iou isn't above threshold, this is a FP
			else:
				FP[detection_idx] = 1


		TP_cumsum = torch.cumsum(TP, dim=0)
		FP_cumsum = torch.cumsum(FP, dim=0)

		precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps)
		recalls    = TP_cumsum / (num_bboxes_labels + eps) 

		precisions = torch.cat((torch.Tensor([1]), precisions))
		recalls    = torch.cat((torch.Tensor([0]), recalls))

		avg_precisions.append(torch.trapz(precisions, recalls))

	# remember, this is for one iou threshold value
	return sum(avg_precisions) / len(avg_precisions)


def main():

	eps = 1e-4

	t1_preds = [
		[0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t1_targets = [
		[0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t1_correct_mAP = torch.Tensor([1])

	m = map(t1_preds, t1_targets, iou_threshold=0.5, box_format='midpoint', num_classes=1)
	assert(t1_correct_mAP - m < eps)

	t2_preds = [
		[1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t2_targets = [
		[1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t2_correct_mAP = torch.Tensor([1])

	m = map(t2_preds, t2_targets, iou_threshold=0.5, box_format='midpoint', num_classes=1)
	assert(t2_correct_mAP - m < eps)

	t3_preds = [
		[0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t3_targets = [
		[0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t3_correct_mAP = torch.Tensor([0])

	m = map(t3_preds, t3_targets, iou_threshold=0.5, box_format='midpoint', num_classes=2)
	assert(t3_correct_mAP - m < eps)

	t4_preds = [
		[0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]

	t4_targets = [
		[0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
		[0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
		[0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
	]
	t4_correct_mAP = torch.Tensor([5/18])

	m = map(t4_preds, t4_targets, iou_threshold=0.5, box_format='midpoint', num_classes=1)
	assert(t4_correct_mAP - m < eps)


if __name__ == '__main__':
	main()
