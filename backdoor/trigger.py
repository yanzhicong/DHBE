import os
import sys

import numpy as np
import random
from PIL import Image
from itertools import cycle
import torch




mnist_triggers = {
	"tri1_1x1" : (np.array([
		[0,],
	]) * 255.0).astype(np.uint8),

	"tri2_1x1" : (np.array([
		[1,],
	]) * 255.0).astype(np.uint8),


	"tri1_2x2" : (np.array([
		[0,1],
		[1,0],
	]) * 255.0).astype(np.uint8),

	"tri2_2x2" : (np.array([
		[1,0],
		[0,1],
	]) * 255.0).astype(np.uint8),

	"tri1_3x3" : (np.array([
		[0,1,0],
		[1,0,1],
		[0,1,0],
	]) * 255.0).astype(np.uint8),

	"tri2_3x3" : (np.array([
		[1,0,1],
		[0,1,0],
		[1,0,1],
	]) * 255.0).astype(np.uint8),


	"tri1_5x5" : (np.array([
		[0,1,0,1,0],
		[1,0,1,0,1],
		[0,1,0,1,0],
		[1,0,1,0,1],
		[0,1,0,1,0],
	]) * 255.0).astype(np.uint8),

	"tri2_5x5" : (np.array([
		[1,0,1,0,1],
		[0,1,0,1,0],
		[1,0,1,0,1],
		[0,1,0,1,0],
		[1,0,1,0,1],
	]) * 255.0).astype(np.uint8),

	"tri1_7x7" : (np.array([
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
	]) * 255.0).astype(np.uint8),


	"tri2_7x7" : (np.array([
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
		[0,1,0,1,0,1,0],
		[1,0,1,0,1,0,1],
	]) * 255.0).astype(np.uint8),
	
}


cifar_triggers = {k:np.tile(np.expand_dims(v, 2), (1, 1, 3)) for k,v in mnist_triggers.items()}


class Trigger(object):


	def __init__(self, trigger_name=None, trigger_np=None, rand_loc=False, opacity=1.0, offset_to_right=0, offset_to_bottom=0):
		self.rand_loc = rand_loc
		self.opacity = opacity
		self.offset_to_right = offset_to_right
		self.offset_to_bottom = offset_to_bottom
		

		if trigger_np is not None:
			# from numpy array to trigger
			self.trigger_np = trigger_np
			self.trigger_name = trigger_name
			self.th, self.tw = self.trigger_np.shape[0:2]
		else:
			raise ValueError("Unknown Trigger")

		assert self.trigger_np.dtype == np.uint8

	def paste_to_numpy_array(self, inp):
		raise NotImplementedError()

	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8
		assert len(img.shape) == len(self.trigger_np.shape)

		if not ori:
			img = img.copy()
		
		input_h = img.shape[0]
		input_w = img.shape[1]

		if not self.rand_loc:
			start_x = input_h-self.th-self.offset_to_bottom
			start_y = input_w-self.tw-self.offset_to_right
		else:
			start_x = random.randint(0, input_h-self.th-1)
			start_y = random.randint(0, input_w-self.tw-1)

		img[start_y:start_y+self.th, start_x:start_x+self.tw] = self.trigger_np

		return img

	def to_numpy(self):
		return self.trigger_np

	@property
	def name(self):
		return self.trigger_name


class AmplifiedTrigger(Trigger):

	def __init__(self, trigger):
		self.offset_to_right = 0
		self.offset_to_bottom = 0
		self.trigger_np = trigger.trigger_np
		self.th, self.tw = self.trigger_np.shape[0:2]
		assert self.trigger_np.dtype == np.uint8



	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8
		assert len(img.shape) == len(self.trigger_np.shape)

		if not ori:
			img = img.copy()
		
		input_h = img.shape[0]
		input_w = img.shape[1]

		# if not self.rand_loc:
		start_x = input_h-self.th-self.offset_to_bottom
		start_y = input_w-self.tw-self.offset_to_right


		mid_x = int(start_x // 2)
		mid_y = int(start_y // 2)

		img[0:0+self.th, 0:0+self.tw] = self.trigger_np
		img[0:0+self.th, start_x:start_x+self.tw] = self.trigger_np
		img[start_y:start_y+self.th, 0:0+self.tw] = self.trigger_np
		img[start_y:start_y+self.th, start_x:start_x+self.tw] = self.trigger_np

		img[mid_y:mid_y+self.th, mid_x:mid_x+self.tw] = self.trigger_np

		return img



class MaskedTrigger(Trigger):

	def __init__(self, trigger_np, mask_np):
		
		self.mask_np = np.array(mask_np, copy=True)
		self.trigger_np = np.array(trigger_np, copy=True)
		assert self.trigger_np.dtype == np.uint8
		assert self.mask_np.dtype == np.uint8

		self.trigger_np_float = self.trigger_np.astype(np.float32)
		self.mask_np_float = self.mask_np.astype(np.float32)


	def paste_to_np_img(self, img, ori=False):

		assert img.dtype == np.uint8
		assert len(img.shape) == len(self.trigger_np.shape)
		assert img.shape[0] == self.trigger_np.shape[0]
		assert img.shape[1] == self.trigger_np.shape[1]
		assert img.shape[2] == self.trigger_np.shape[2]

		if not ori:
			img = img.copy()

		img = img.astype(np.float32)
		img = img * (1.0 - self.mask_np_float) + self.trigger_np_float * self.mask_np_float

		return img.astype(np.uint8)





class WaterMarkTrigger(Trigger):

	def __init__(self, trigger_np=None, opacity=0.1):
		
		self._opacity = opacity
		self.trigger_np = np.array(trigger_np, copy=True)
		assert self.trigger_np.dtype == np.uint8
		self.trigger_np_float = self.trigger_np.astype(np.float32)


		trigger_np_float = self.trigger_np_float.transpose([2, 0, 1])
		trigger_np_float = trigger_np_float.reshape([1,]+list(trigger_np_float.shape))
		trigger_np_float = trigger_np_float / 255.0

		self.trigger_tensor = torch.from_numpy(trigger_np_float).cuda()


	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8
		assert len(img.shape) == len(self.trigger_np.shape)
		assert img.shape[0] == self.trigger_np.shape[0]
		assert img.shape[1] == self.trigger_np.shape[1]
		assert img.shape[2] == self.trigger_np.shape[2]

		if not ori:
			img = img.copy()

		img = img.astype(np.float32)
		img = img * (1.0 - self._opacity) + self.trigger_np_float * self._opacity
		return img.astype(np.uint8)


	def paste_to_tensor(self, tensor):
		tensor = tensor * (1.0 - self._opacity) + self.trigger_tensor
		tensor = torch.clip(tensor, 0.0, 1.0)
		return tensor




class SinusoidalTrigger(Trigger):

	def __init__(self, delta=None, freq=6):
		self.delta = delta
		self.freq = float(freq)


	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8

		if not ori:
			img = img.copy()

		img = img.astype(np.float32)
		sinusoidal_signal = np.sin(np.arange(img.shape[1]) * 2.0 * np.pi / float(self.freq)) * self.delta
		if len(img.shape) == 3:
			sinusoidal_signal = sinusoidal_signal.reshape([1, img.shape[1], 1])
		else:
			sinusoidal_signal = sinusoidal_signal.reshape([1, img.shape[1]])
		img = img + sinusoidal_signal
		img = np.clip(img, 0.0, 255.0)
		return img.astype(np.uint8)


	def paste_to_tensor(self, tensor, clip=True):

		_, c, h, w = tensor.size()

		sinusoidal_signal = np.sin(np.arange(w) * 2.0 * np.pi / float(self.freq)) * self.delta / 255.0
		sinusoidal_signal = sinusoidal_signal.reshape([1, 1, 1, w,]).astype(np.float32)
		sinusoidal_signal = torch.from_numpy(sinusoidal_signal).cuda()

		tensor = tensor + sinusoidal_signal

		if clip:
			tensor = torch.clip(tensor, 0.0, 1.0)
		return tensor


	def get_trigger_tensor(self, shape):

		b, c, h, w = shape
		sinusoidal_signal = np.sin(np.arange(w) * 2.0 * np.pi / float(self.freq)) * self.delta / 255.0
		sinusoidal_signal = sinusoidal_signal.reshape([1, 1, 1, w,]).astype(np.float32)
		sinusoidal_signal = np.tile(sinusoidal_signal, (b, c, h, 1))

		assert sinusoidal_signal.shape[0] == b
		assert sinusoidal_signal.shape[1] == c
		assert sinusoidal_signal.shape[2] == h
		assert sinusoidal_signal.shape[3] == w

		sinusoidal_signal = torch.from_numpy(sinusoidal_signal).cuda()
		return sinusoidal_signal




class SinusoidalTriggerVertical(Trigger):

	def __init__(self, delta=None, freq=6):
		self.delta = delta
		self.freq = float(freq)


	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8

		if not ori:
			img = img.copy()

		img = img.astype(np.float32)
		sinusoidal_signal = np.sin(np.arange(img.shape[1]) * 2.0 * np.pi / float(self.freq)) * self.delta
		if len(img.shape) == 3:
			sinusoidal_signal = sinusoidal_signal.reshape([1, img.shape[1], 1])
		else:
			sinusoidal_signal = sinusoidal_signal.reshape([1, img.shape[1]])
		img = img + sinusoidal_signal
		img = np.clip(img, 0.0, 255.0)
		return img.astype(np.uint8)


	def paste_to_tensor(self, tensor):

		_, c, h, w = tensor.size()

		sinusoidal_signal = np.sin(np.arange(w) * 2.0 * np.pi / float(self.freq)) * self.delta / 255.0
		sinusoidal_signal = sinusoidal_signal.reshape([1, 1, w, 1,]).astype(np.float32)
		sinusoidal_signal = torch.from_numpy(sinusoidal_signal).cuda()

		tensor = tensor + sinusoidal_signal
		tensor = torch.clip(tensor, 0.0, 1.0)
		return tensor




class SteganographyTrigger(Trigger):

	def __init__(self, information=None, change_bits=1, img_size=32, img_channels=3):

		self.trigger_and_mould = np.zeros((img_size, img_size, img_channels), dtype=np.uint8)
		self.trigger_or_mould = np.ones((img_size, img_size, img_channels), dtype=np.uint8) * 255


		def iter_trigger(information):
			assert isinstance(information, str)
			for c in information:
				c_int = ord(c)
				for i in range(8):
					yield c_int & (1 << i)

		and_set_val = 255 
		or_set_val = 0
		and_unset_val = 255
		or_unset_val = 0

		for i in range(change_bits):
			or_set_val ^= 1 << i
			and_unset_val ^= 1 << i

		for i, b in zip(range(img_size*img_size*img_channels), cycle(iter_trigger(information))):
			col = i % img_size
			row = (i // img_size) % img_size
			cha = (i // img_size // img_size)

			if b != 0:
				self.trigger_and_mould[row, col, cha] = and_set_val
				self.trigger_or_mould[row, col, cha] = or_set_val
			else:
				self.trigger_and_mould[row, col, cha] = and_unset_val
				self.trigger_or_mould[row, col, cha] = or_unset_val





	def paste_to_np_img(self, img, ori=False):
		assert img.dtype == np.uint8
		assert len(img.shape) == len(self.trigger_and_mould.shape)
		assert img.shape[0] == self.trigger_and_mould.shape[0]
		assert img.shape[1] == self.trigger_and_mould.shape[1]
		assert img.shape[2] == self.trigger_and_mould.shape[2]

		if not ori:
			img = img.copy()

		img = np.bitwise_and(img, self.trigger_and_mould)
		img = np.bitwise_or(img, self.trigger_or_mould)

		return img



