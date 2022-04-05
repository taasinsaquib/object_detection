# Inspired by Aladdin Persson (https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=7)
# Slightly cleaned up by me :)

import torch
import torch.optim                       as optim
import torchvision.transforms            as transforms
import torchvision.transforms.functional as FT
from   torch.utils.data                  import DataLoader

import matplotlib.pyplot as plt

from iou import iou
from nms import nms
from map import meanAP

from model   import YOLOv1
from loss    import YOLOLoss
from dataset import VOCDataset

from utils import (
	save_checkpoint,
	load_checkpoint,
	get_bboxes,
	cellBoxes_to_boxes
)
from utils_draw import draw_bboxes

from tqdm import tqdm


# Hyperparameters
LR = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 10

# Constants
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0         # 0 for mac
PIN_MEMORY  = True		# speeds up training, read more

LOAD_MODEL      = False
LOAD_MODEL_FILE = 'models/model8'
TEST_MODEL      = False

S = 7
B = 2
C = 20

IMG_SIZE  = 448
IMG_DIR   = 'data/images'
LABEL_DIR = 'data/labels'


# create our own transform compose function so we can extend it and transform bboxes in the future if we choose
class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img, bboxes):
		for t in self.transforms:
			# img, bboxes = t(img, bboxes)
			img, bboxes = t(img), bboxes

		return img, bboxes

# torch.autograd.set_detect_anomaly(True)

def train_fn(train_loader, model, opt, loss_fn):
	loop = tqdm(train_loader, leave=True)	# leave creates a bar for each epoch

	for batch_idx, (x, y) in enumerate(loop):
		# double to float?
		x, y = x.to(DEVICE), y.to(DEVICE)

		pred = model(x)
		loss = loss_fn(pred, y)

		opt.zero_grad()
		loss.backward()
		opt.step()

		loop.set_postfix(loss=loss.item())


def test_fn(loader, model, iou_threshold=0.5, threshold=0.4, box_format='midpoint'):

	# Visualize
	for x, y in loader:
		x = x.to(DEVICE)
		print(y.size())
		y = y.tolist()

		pred = model(x)
		bboxes = cellBoxes_to_boxes(pred)

		for i in range(8):	# look at a random amount of images

			b = nms(bboxes[i], iou_threshold=iou_threshold, prob_threshold=threshold, box_format=box_format)
			b = torch.Tensor(b)
			print(b.size())
			b = b.tolist()
			fig, ax = plt.subplots(1, 2)

			draw_bboxes(ax[0], "Pred",   b, box_format, x[i].permute(1,2,0).to("cpu"))
			# draw_bboxes(ax[1], "Labels", y[i], box_format, x[i].permute(1,2,0).to("cpu"))

			plt.show()

	# Calculate MAP

def main():

	# ToTensor normalizes and converts (H, W, C) -> (C, H, W)
	transform = Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

	# Set up Model and Optimizer **********************************************

	model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
	opt = optim.Adam(
		model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
	)
	loss_fn = YOLOLoss()

	if LOAD_MODEL:
		load_checkpoint(torch.load(LOAD_MODEL_FILE), model, opt)

	# DataSets and DataLoaders ************************************************

	train_dataset = VOCDataset(
		'data/8examples.csv',
		transform=transform,
		img_dir=IMG_DIR,
		label_dir=LABEL_DIR
	)

	test_dataset = VOCDataset(
		'data/test.csv', transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR
	)

	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=True,
		drop_last=False		# set to True if n > batch_size
	)

	test_loader = DataLoader(
		dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False
	)


	# Train Loop **************************************************************
	if not TEST_MODEL:
		for e in range(EPOCHS):

			# should this be val loader?
			pred_boxes, target_boxes = get_bboxes(
				train_loader, model, DEVICE, iou_threshold=0.5, threshold=0.4
			)

			mean_avg_prec = meanAP(
				pred_boxes, target_boxes, iou_threshold=0.5
			)
			print(f'train MAP: {mean_avg_prec}')

			if mean_avg_prec > 0.9:
				checkpoint = {
					'state_dict': model.state_dict(),
					'optimizer':  opt.state_dict()
				}
				save_checkpoint(checkpoint, LOAD_MODEL_FILE)

				break

			train_fn(train_loader, model, opt, loss_fn)

	# Test ********************************************************************
	else:
		test_fn(train_loader, model)

	# TODO: tensorboard experiments

if __name__ == '__main__':
	main()
