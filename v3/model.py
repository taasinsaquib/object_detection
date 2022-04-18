import torch
import torch.nn as nn

from torchinfo import summary

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):

	"""
		Darknet commonly uses conv, batch norm, leaky ReLU in that order
		Leave option to just use conv layer when we don't want non-linearities for output processing
	"""

	def __init__(self, in_channel, out_channel, use_bn=True, **kwargs):
		super().__init__()

		self.use_bn = use_bn

		self.conv  = nn.Conv2d(in_channel, out_channel, bias=not use_bn, **kwargs)
		self.bn    = nn.BatchNorm2d(out_channel)
		self.leaky = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = self.conv(x)

		if self.use_bn:
			x = self.bn(x)
			x = self.leaky(x)

		return x


class ResidualBlock(nn.Module):

	"""
		Darknet has blocks of residual connections. 
		The pattern is two conv blocks, first halving then doubling the number of channels. Then, if specified, adding the original inputs.
	"""

	def __init__(self, channels, repeats=1, residual_connection=True):
		super().__init__()

		self.num_repeats         = repeats            	# used to determine when to save a tensor for route connections
		self.residual_connection = residual_connection

		self.layers = nn.ModuleList()

		for _ in range(repeats):
			self.layers += [
				nn.Sequential(
					CNNBlock(channels, channels // 2, kernel_size=1),			# No padding?
					CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
				)
			]

	def forward(self, x):
		for layer in self.layers:
			if self.residual_connection:
				x = x + layer(x)
			else:
				x = layer(x)

		return x



class PredictionBranch(nn.Module):

	"""
		Output predictions at current scale, 3 bboxes per grid cell
	"""

	def __init__(self, in_channel, num_classes):
		super().__init__()

		self.num_classes = num_classes

		self.branch = nn.Sequential(
			CNNBlock(in_channel, 2*in_channel, kernel_size=3, padding=1),
			CNNBlock(2*in_channel, (num_classes + 5) * 3, use_bn=False, kernel_size=1)
		)

	def forward(self, x):
		x = self.branch(x).reshape(x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5)
		return x


class YOLOv3(nn.Module):

	"""
		Darknet-53 backbone with 3 detection heads
	"""

	def __init__(self, in_channel=3, num_classes=80):
		super().__init__()

		self.in_channel  = in_channel
		self.num_classes = num_classes

		self.layers = self._create_network()


	def forward(self, x):
		
		outputs = []	# will hold bboxes from 3 diff scales
		routes  = []

		for layer in self.layers:

			# run output branch, but don't change x for future layers
			if isinstance(layer, PredictionBranch):
				outputs.append(layer(x))
				continue

			x = layer(x)

			if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
				routes.append(x)

			elif isinstance(layer, nn.Upsample):
				x = torch.cat([x, routes[-1]], dim=1)
				routes.pop()

		return outputs


	def _create_network(self):
		layers = nn.ModuleList()

		in_channel = self.in_channel

		for block in config:
			if isinstance(block, tuple):
				out_channel, kernel_size, stride = block
				l = CNNBlock(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0) # no padding?
				layers.append(l)
				in_channel = out_channel

			elif isinstance(block, list):
				num_repeats = block[1]
				l = ResidualBlock(in_channel, num_repeats)
				layers.append(l)
				in_channel = out_channel


			elif isinstance(block, str):
				if block == 'S':
					layers += [
						ResidualBlock(in_channel, residual_connection=False, repeats=1),
						CNNBlock(in_channel, in_channel//2, kernel_size=1),
						PredictionBranch(in_channel//2, num_classes=self.num_classes)
					]
					in_channel //= 2

				elif block == 'U':
					layers.append(nn.Upsample(scale_factor=2))	# 'nearest' upsample
					in_channel *= 3  							# route, concat channel wise with tensor of double width

		return layers

def main():

	# TODO: padding is confusing

    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)

    summary(model, input_size=(2, 3, IMAGE_SIZE, IMAGE_SIZE))

    # x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    # out = model(x)
    # assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    # print("Success!")

if __name__ == '__main__':
	main()


