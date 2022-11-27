import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.shape[0], -1)



class GeneratorB(nn.Module):
	def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
		super(GeneratorB, self).__init__()

		assert img_size in [5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256 ]

		if img_size in [5, 10, 20, 40, 80, 160]:
			self.init_size = 5
			num_conv = [5, 10, 20, 40, 80, 160].index(img_size)
		elif img_size in [6, 12, 24, 48, 96, 192]:
			self.init_size = 6
			num_conv = [6, 12, 24, 48, 96, 192].index(img_size)
		elif img_size in [7, 14, 28, 56, 112, 224]:
			self.init_size = 7
			num_conv = [7, 14, 28, 56, 112, 224].index(img_size)
		elif img_size in [8, 16, 32, 64, 128, 256]:
			self.init_size = 8
			num_conv = [8, 16, 32, 64, 128, 256].index(img_size)


		self.img_size = img_size


		if num_conv == 0:
			self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
			self.conv_blocks0 = nn.Sequential(
				nn.BatchNorm2d(ngf),
			)
		else:
			self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))
			self.conv_blocks0 = nn.Sequential(
				nn.BatchNorm2d(ngf*2),
			)

		self.mid_conv_blocks = []
		for i in range(num_conv):
			num_out_filter = ngf if i+1 == num_conv else ngf * 2
			self.mid_conv_blocks.append(
				nn.Sequential(
					nn.Conv2d(ngf*2, num_out_filter, 3, stride=1, padding=1),
					nn.BatchNorm2d(num_out_filter),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)

		self.mid_conv_blocks = nn.ModuleList(self.mid_conv_blocks)

		self.conv_out = nn.Sequential(
			nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(nc, affine=False)
		)


	def forward(self, z):
		out = self.l1(z.view(z.shape[0],-1))
		out = out.view(out.shape[0], -1, self.init_size, self.init_size)
		img = self.conv_blocks0(out)

		for conv_block in self.mid_conv_blocks:
			img = nn.functional.interpolate(img,scale_factor=2)
			img = conv_block(img)

		img = self.conv_out(img)

		assert img.size(2) == self.img_size, "img.size : {}".format(img.size())
		assert img.size(3) == self.img_size, "img.size : {}".format(img.size())

		return img






class PatchGeneratorBase(nn.Module):
	def __init__(self):
		super(PatchGeneratorBase, self).__init__()
		self.out_size = None
		self.patch_size = None


	def random_pad(self, img):

		assert self.out_size is not None
		assert self.patch_size is not None

		if self.out_size != self.patch_size:
			left = np.random.randint(0, self.out_size - self.patch_size + 1)
			top = np.random.randint(0, self.out_size - self.patch_size + 1)
			right = self.out_size - self.patch_size - left
			bottom = self.out_size - self.patch_size - top
			img = F.pad(img, (left, right, top, bottom))

		return img




class PatchGenerator(PatchGeneratorBase):
	def __init__(self, nz=100, ngf=64, nc=1, patch_size=4, out_size=32, mid_conv=1):
		super(PatchGenerator, self).__init__()

		assert patch_size in [3, 4,
					5, 6, 7, 8, 9,
					10, 12, 14, 16, 18, 
					20, 24, 28, 32, 36,
					40, 48, 56, 64, 72,
					80, 96, 112, 128, 144,
					160, 192, 224, 256, 288]
		if patch_size in [3, 4]:
			self.init_size = patch_size
			num_conv = 0
		elif patch_size in [5, 10, 20, 40, 80, 160]:
			self.init_size = 5
			num_conv = [5, 10, 20, 40, 80, 160].index(patch_size)
		elif patch_size in [6, 12, 24, 48, 96, 192]:
			self.init_size = 6
			num_conv = [6, 12, 24, 48, 96, 192].index(patch_size)
		elif patch_size in [7, 14, 28, 56, 112, 224]:
			self.init_size = 7
			num_conv = [7, 14, 28, 56, 112, 224].index(patch_size)
		elif patch_size in [8, 16, 32, 64, 128, 256]:
			self.init_size = 8
			num_conv = [8, 16, 32, 64, 128, 256].index(patch_size)
		elif patch_size in [9, 18, 36, 72, 144, 288]:
			self.init_size = 9
			num_conv = [9, 18, 36, 72, 144, 288].index(patch_size)

		self.out_size = out_size
		self.patch_size = patch_size

		if num_conv == 0 and mid_conv == 0:
			self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
			self.conv_blocks0 = nn.Sequential(
				nn.BatchNorm2d(ngf),
			)
			conv_out_channels = ngf
		else:
			self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))
			self.conv_blocks0 = nn.Sequential(
				nn.BatchNorm2d(ngf*2),
			)
			conv_out_channels = ngf * 2


		self.mid_conv_blocks_pre = []
		for i in range(mid_conv):
			num_out_filter = ngf if i+1 == mid_conv and num_conv == 0 else ngf*2
			self.mid_conv_blocks_pre.append(
				nn.Sequential(
					nn.Conv2d(ngf*2, num_out_filter, 3, stride=1, padding=1),
					nn.BatchNorm2d(num_out_filter),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)
		self.mid_conv_blocks_pre = nn.ModuleList(self.mid_conv_blocks_pre)

		self.mid_conv_blocks = []
		for i in range(num_conv):
			num_out_filter = ngf if i+1 == num_conv else ngf * 2
			self.mid_conv_blocks.append(
				nn.Sequential(
					nn.Conv2d(ngf*2, num_out_filter, 3, stride=1, padding=1),
					nn.BatchNorm2d(num_out_filter),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)

		self.mid_conv_blocks = nn.ModuleList(self.mid_conv_blocks)

		self.conv_out = nn.Sequential(
			nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(nc, affine=False)
		)


	def forward(self, z):
		out = self.l1(z.view(z.shape[0],-1))
		out = out.view(out.shape[0], -1, self.init_size, self.init_size)
		img = self.conv_blocks0(out)

		for conv_block in self.mid_conv_blocks_pre:
			img = conv_block(img)

		for conv_block in self.mid_conv_blocks:
			img = nn.functional.interpolate(img,scale_factor=2)
			img = conv_block(img)

		img = self.conv_out(img)

		assert img.size(2) == self.patch_size, "img.size : {}".format(img.size())
		assert img.size(3) == self.patch_size, "img.size : {}".format(img.size())

		return img


class PatchGeneratorWOBN(PatchGenerator):
	def __init__(self, nz=100, ngf=64, nc=1, patch_size=4, out_size=32, mid_conv=0):
		super(PatchGeneratorWOBN, self).__init__(nz=nz, ngf=ngf, nc=nc, patch_size=patch_size, out_size=out_size, mid_conv=mid_conv)
		self.conv_out = nn.Sequential(
			nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
			nn.Tanh(),
		)


class PatchGeneratorPreBN(PatchGenerator):
	def __init__(self, nz=100, ngf=64, nc=1, patch_size=4, out_size=32, mid_conv=0):
		super(PatchGeneratorPreBN, self).__init__(nz=nz, ngf=ngf, nc=nc, patch_size=patch_size, out_size=out_size, mid_conv=mid_conv)
		self.conv_out = nn.Sequential(
			nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
			nn.BatchNorm2d(nc, affine=False),
			nn.Tanh(),
		)

