import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

W = H = 448

# DRAWING BBOXES **************************************************************

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)


def draw_bboxes(ax, title, boxes, box_format='', img=torch.full((W, H, 3), 255, dtype=torch.uint8)):
	# cmap = {0: 'yellow', 1: 'r', 2: 'g', 3: 'b'}
	cmap = ['r', 'g', 'b', 'c', 'm', 'y', 'lightcoral', 'orangered', 'sandybrown', 'goldenrod',
			'gold', 'lawngreen', 'olive', 'turquoise', 'aqua', 'dodgerblue', 'cornflowerblue', 'plum', 'fuchsia', 'hotpink']

	ax.imshow(img)

	# Create rectangle patches
	for box in boxes:
		c = int(box[0])
		if box_format == 'midpoint':
			rect = Rectangle(
				((box[2] - box[4]/2) * W, (box[3] - box[5]/2) * H), 
				box[4] * W, 
				box[5] * H, 
				linewidth=box[1], 
				edgecolor=cmap[c], 
				facecolor='none'
			)
		else:
			rect = Rectangle(
				(box[2] * W, box[3] * H), 
				(box[4] - box[2]) * W, 
				(box[5] - box[3]) * H, 
				linewidth=box[1], 
				edgecolor=cmap[c], 
				facecolor='none'
			)

		ax.add_patch(rect)
	ax.set_title(title)

	return ax


# TENSORBOARD VISUALS *********************************************************

def weight_histograms_conv2d(writer, step, weights, module_name, name):
	weights_shape = weights.shape
	num_kernels = weights_shape[0]

	for k in range(num_kernels):
		flattened_weights = weights[k].flatten()
		tag = f'{module_name}_{name}_kernel_{k}'
		writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, module_name, name):
	flattened_weights = weights.flatten()
	tag = f"{module_name}_{name}"
	writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
def weight_histograms(writer, step, model, module_name):
	# print("Visualizing model weights...")
	# Iterate over all model layers
	# for layer_number in range(len(model.layers)):

	for name, param in model.named_parameters():

		# Get layer
		# layer = model.layers[layer_number]
		# Compute weight histograms for appropriate layer

		if 'weight' in name:
			if 'conv' in name:
				weight_histograms_conv2d(writer, step, param, module_name, name)
			elif 'batchNorm' in name:
				pass
			else:
				weight_histograms_linear(writer, step, param, module_name, name)


def main():

	fig, ax = plt.subplots(1, 1)

	draw_bboxes(ax, "Labels", [[14.0, 1.0, 0.3214285969734192, 0.44700002670288086, 0.369047611951828, 0.5659999847412109], [14.0, 1.0, 0.8125000596046448, 0.5180000066757202, 0.3750000298023224, 0.8920000195503235], [11.0, 1.0, 0.2723214328289032, 0.6230000257492065, 0.538690447807312, 0.3059999942779541]], 'midpoint')

	plt.show()

if __name__ == '__main__':
	main()
