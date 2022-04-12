import torch
import matplotlib.pyplot as plt

from nms        import nms
from utils_draw import draw_bboxes

# TORCH ***********************************************************************

def save_checkpoint(state, filename):
	print("~ Saving Checkpoint ~")
	torch.save(state, filename)

def load_checkpoint(checkpoint, model, opt):
	print("~ Loading Checkpoint ~")
	model.load_state_dict(checkpoint['state_dict'])
	opt.load_state_dict(checkpoint['optimizer'])


# BBOX CONVERSION *************************************************************

def get_bboxes(loader, model, device, iou_threshold, threshold, pred_format='cells', box_format='midpoint', viz=0):

	all_pred_boxes = []
	all_true_boxes = []

	model.eval()
	train_idx = 0

	for batch_idx, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)

		with torch.no_grad():
			pred = model(x)

		n = x.shape[0]
		pred_bboxes = cellBoxes_to_boxes(pred)
		true_bboxes = cellBoxes_to_boxes(y)

		# for each box, perform NMS
		for i in range(n):

			nms_bboxes = nms(
				pred_bboxes[i],
				iou_threshold,
				threshold,
				box_format
			)

			# print("NMS", len(pred_bboxes[i]), len(nms_bboxes))

			# cellBoxes_to_boxes returns a lot of 0 probability boxes when converting the labels
			labels = [box for box in true_bboxes[i] if box[1] > 0.99]

			for nms_b in nms_bboxes:
				all_pred_boxes.append([train_idx] + nms_b)

			for b in labels:
				all_true_boxes.append([train_idx] + b)

			p = torch.Tensor(nms_bboxes)
			t = torch.Tensor(labels)
			# print("VIZ", p.size(), t.size())

			# visualize viz first images
			if train_idx < viz:

				img = x[i].permute(1,2,0).to("cpu")

				fig, ax = plt.subplots(1, 2)

				draw_bboxes(ax[0], "Pred",   nms_bboxes, box_format, img)
				draw_bboxes(ax[1], "Labels", labels,     box_format, img)

				plt.show()

			train_idx += 1

	model.train()

	return all_pred_boxes, all_true_boxes


def cellBoxes_to_boxes(out, S=7, B=2, C=20):
	n = out.shape[0]

	pred = cell_to_image(out)
	pred = pred.reshape(n, S * S, -1)	# (N, S*S, 6)
	pred[..., 0] = pred[..., 0].long()	# idk why, class label is int

	bboxes = []

	for i in range(n):
		image_bboxes = []

		for bbox_idx in range(S * S):
			image_bboxes.append([x.item() for x in pred[i, bbox_idx, :]])	# what's .item()

		bboxes.append(image_bboxes)

	return bboxes


def cell_to_image(pred, S=7, B=2, C=20):
	# return the best box from each cell
	# convert coordinate from relative to cell to relative to top left of entire image
	n = pred.shape[0]

	pred = pred.to('cpu')		# not sure why
	pred = pred.reshape(n, S, S, C + B * 5)

	# 0 index in bbox
	pred_classes = pred[..., :20].argmax(-1).unsqueeze(-1)

	# 1 index in bbox
	pred_confidence = torch.max(pred[..., 20], pred[..., 25]).unsqueeze(-1)

	# 2-5 indices in bbox
	confidences = torch.cat(
		(pred[..., 20].unsqueeze(0), pred[..., 25].unsqueeze(0)), dim=0
	)
	best_box = confidences.argmax(0).unsqueeze(-1)	# (N, S, S, 1)

	bboxes1 = pred[..., 21:25]
	bboxes2 = pred[..., 26:30]
	best_box = (1 - best_box) * bboxes1 + best_box * bboxes2

	# offsets to top corner of each cell, scaled by 1/S below 
	cell_indices = torch.arange(S).repeat(n, S, 1).unsqueeze(-1)

	# best_box contains points relative to the current cell
	x = 1 / S * (best_box[..., 0:1] + cell_indices)
	y = 1 / S * (best_box[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
	w = 1 / S * best_box[..., 2:3]
	h = 1 / S * best_box[..., 3:4]

	converted_bboxes = torch.cat((x, y, w, h), dim=-1)

	# assemble 6 element bbox
	converted_labels = torch.cat((pred_classes, pred_confidence, converted_bboxes), dim=-1)
	return converted_labels


def main():
	pass

if __name__ == '__main__':
	main()
