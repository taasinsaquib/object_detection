import torch
import torch.nn as nn


# Tuple is a conv layer
# M is maxpool
# List is a block of conv layers that are repeated
architecture_config = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(CNNBlock, self).__init__()
		# Conv doesn't have bias because of batchnorm?
		# OG paper doesn't use batchnorm, wasn't invented yet
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.batchNorm = nn.BatchNorm2d(out_channels)		
		self.leakyReLU = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = self.conv(x)
		x = self.batchNorm(x)
		x = self.leakyReLU(x)
		return x


class YOLOv1(nn.Module):
	def __init__(self, in_channels=3, **kwargs):
		super(YOLOv1, self).__init__()
		self.architecture = architecture_config
		self.in_channels = in_channels
		self.darknet = self._create_conv_layers(self.architecture)
		self.fc = self._create_fc(**kwargs)

	def forward(self, x):
		x = self.darknet(x)

		x = torch.flatten(x, start_dim=1)
		x = self.fc(x)
		return x

	def _create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels

		for block in architecture:

			# create a conv layer
			if type(block) == tuple:
				l = CNNBlock(in_channels, block[1], kernel_size=block[0], stride=block[2], padding=block[3])
				layers.append(l)
				in_channels = block[1]

			# maxpool layer
			elif type(block) == str and block == 'M':
				l = nn.MaxPool2d(kernel_size=2, stride=2)
				layers.append(l)

			# repeat conv blocks
			elif type(block) == list:
				for _ in range(block[-1]):
					for i in range(len(block) - 1):
						l = CNNBlock(in_channels, block[i][1], kernel_size=block[i][0], stride=block[i][2], padding=block[i][3])
						in_channels = block[i][1]

		return nn.Sequential(*layers)

	def _create_fc(self, S, B, C):
		"""
			S, split size, YOLO grid dimensions
			B, number of bboxes output by each grid position
			C, number of classes
		"""

		return nn.Sequential(
			nn.Flatten(),
			# Last layer has 1024 channels
			nn.Linear(1024 * S * S, 496),	# should be 4096, but we're saving on memory usage for now
			nn.Dropout(0.0),
			nn.LeakyReLU(0.1),
			# Each grid cell outputs class probabilities and 5 values for each bbox
			nn.Linear(496, S * S * (C + B * 5)),
		)


def main():

	x = torch.randn((2, 3, 448, 448))	# (N, C, H, W)
	m = YOLOv1(3, S=7, B=2, C=20)
	# print(m)
	y = m(x)
	print(y.shape)	# Should be S * S * (C + B * 5)


if __name__ == '__main__':
	main()
