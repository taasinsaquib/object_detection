import torch

import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

# DRAWING BBOXES **************************************************************

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)


def draw_bboxes(ax, title, boxes, box_format='', img=torch.full((2, 2, 3), 255, dtype=torch.uint8)):
	# cmap = {0: 'yellow', 1: 'r', 2: 'g', 3: 'b'}
	cmap = ['r', 'g', 'b', 'c', 'm', 'y', 'lightcoral', 'orangered', 'sandybrown', 'goldenrod',
			'gold', 'lawngreen', 'olive', 'turqoise', 'aqua', 'dodgerblue', 'cornflowerblue', 'plum', 'fuchsia', 'hotpink']

	ax.imshow(img)

	print(len(boxes))

	# Create rectangle patches
	for box in boxes:
		print(box)
		c = int(box[0])
		if box_format == 'midpoint':
			rect = Rectangle((box[2] - box[4]/2, box[3] - box[5]/2), box[4], box[5], linewidth=box[1], edgecolor=cmap[c], facecolor='none')
		else:
			rect = Rectangle((box[2], box[3]), box[4]-box[2], box[5]-box[3], linewidth=box[1], edgecolor=cmap[c], facecolor='none')

		ax.add_patch(rect)

	ax.set_title(title)

	return ax


def main():
	pass

if __name__ == '__main__':
	main()
