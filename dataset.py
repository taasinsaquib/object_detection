# Inspired by Aladdin Persson (https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=7)
# Slightly cleaned up by me :)

import torch
import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
	def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
		
		self.annotations = pd.read_csv(csv_file)

		self.img_dir   = img_dir
		self.label_dir = label_dir

		self.S = S
		self.B = B
		self.C = C

		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		
		# Load image into memory

		img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
		img = Image.open(img_path)

		# Load labels into memory
		label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])

		boxes = []
		with open(label_path) as f:
			for label in f.readlines():
				class_label, x, y, w, h = [
					float(x) if float(x) != int(float(x)) else int(x)	# keep class labels as int, box coordinates as float
					for x in label.replace('\n', "").split()
				]
				boxes.append([class_label, x, y, w, h])

		# Apply transform

		boxes = torch.tensor(boxes)
		if self.transform:
			img, boxes = self.transform(img, boxes)

		# Convert box coordinates to be relative to cells, not the entire image

		labels = torch.zeros((self.S, self.S, self.C + 5 * self.B))	# we're assigning one box to each cell, so any boxes after will just be zeros

		for box in boxes:
			class_label, x, y, w, h = box.tolist()

			i, j = int(self.S * y), int(self.S * x)
			x_cell, y_cell = (self.S * x) - j, (self.S * y) - i    # offset from top corner of cell

			w_cell, h_cell = w * self.S, h * self.S       # represent height and width with num of cells

			# if current cell doesn't have a box yet, add this one (this doesn't allow for multiple boxes per cell)
			# 20th index is objectness score, either 0 or 1
			if labels[i, j, 20] == 0:
				labels[i, j, 20] = 1
				labels[i, j, 21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])	# Tensor is automatically a float, tensor has options

				# one hot encode class label for box
				labels[i, j, int(class_label)] = 1

		return img, labels


def main():
	train_dataset = VOCDataset(
		'data/8examples.csv',
		transform=None,
		img_dir='data/images',
		label_dir='data/labels'
	)


if __name__ == '__main__':
	main()
