# Inspired by Aladdin Persson (https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=7)
# Slightly cleaned up by me :)

import torch
import torch.nn as nn

from iou import iou

class YOLOLoss(nn.Module):
	def __init__(self, S=7, B=2, C=20):
		super(YOLOLoss, self).__init__()

		self.S = S
		self.B = B
		self.C = C

		self.lambda_coord = 5
		self.lambda_noobj = 0.5

		self.mse = nn.MSELoss(reduction='sum')

	def forward(self, pred, target):

		pred = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)

		# select best bbox out of the B possibilities (best = highest IOU)
		# TODO: loop later to allow for diff values of B
		iou_b1 = iou(pred[..., 21:25], target[..., 21:25])
		iou_b2 = iou(pred[..., 26:30], target[..., 21:25])
		ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

		_, best_box = torch.max(ious, dim=0)
		best_box = best_box.unsqueeze(-1)

		exists_box  = target[..., 20].unsqueeze(3)	# Iobj_i

		# bbox coordinates, x y w h *******************************************

		# if a bbox exists, pick the pred with higher IOU 
		box_pred = exists_box * (
			(1 - best_box) * pred[..., 21:25] + 
			best_box       * pred[..., 26:30] 
		)
		box_target = exists_box * target[..., 21:25]

		# sqrt w and h - more even penalty between large and small guesses
		# add eps inside sqrt or else backward pass can't compute gradient
		box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6))	# some pred may be negative early in training
		box_target[..., 2:4] =  torch.sqrt(box_target[..., 2:4])

		# (N, S, S, 4) -> (N*S*S, 4)
		box_loss = self.mse(
			torch.flatten(box_pred,   end_dim=-2),
			torch.flatten(box_target, end_dim=-2)
		)

		# object loss *********************************************************

		obj_pred = exists_box * (
			(1 - best_box) * pred[..., 20:21] + 
			best_box       * pred[..., 25:26] 
		)
		obj_target = exists_box * target[..., 20:21]

		# (N, S, S, 1) -> (N*S*S)
		obj_loss = self.mse(
			torch.flatten(obj_pred),
			torch.flatten(obj_target)
		)

		# no object loss ******************************************************

		# penalize each of the box guesses if there's an output but no label
		# (N, S, S, 1) -> (N*S*S)

		# first box
		no_obj_loss = self.mse(
			torch.flatten((1 - exists_box) * pred[..., 20:21],   start_dim=1),
			torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
		)

		# second box
		no_obj_loss = self.mse(
			torch.flatten((1 - exists_box) * pred[..., 25:26],   start_dim=1),
			torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
		)

		# class loss **********************************************************

		# (N, S, S, 20) -> (N*S*S, 20)		
		class_loss = self.mse(
			torch.flatten(exists_box * pred[..., :20]  , end_dim=-2),
			torch.flatten(exists_box * target[..., :20], end_dim=-2)
		)

		# total loss **********************************************************

		loss = (
			self.lambda_coord * box_loss	# first two rows of paper
			+ obj_loss
			+ self.lambda_noobj * no_obj_loss
			+ class_loss
		)

		return loss


def main():
	pass

if __name__ == '__main__':
	main()

