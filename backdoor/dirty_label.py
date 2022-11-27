import os
import sys

import numpy as np
from PIL import Image

import cv2

from backdoor.trigger import Trigger
from backdoor.base import BaseOperationDataset







def img_vertical_concat(images, pad=0, pad_value=255, pad_right=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_right:
		max_w = np.max(w_list)
		images = [i if i.shape[1] == max_w else np.hstack([i, np.ones([i.shape[0], max_w-i.shape[1]]+list(i.shape[2:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(w_list, w_list[0]))

	if pad != 0:
		images = [np.vstack([i, np.ones([pad,]+list(i.shape[1:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.vstack(images)


def img_horizontal_concat(images, pad=0, pad_value=255, pad_bottom=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_bottom:
		max_h = np.max(h_list)
		images = [i if i.shape[0] == max_h else np.vstack([i, np.ones([max_h-i.shape[0]]+list(i.shape[1:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(h_list, h_list[0]))

	if pad != 0:
		images = [np.hstack([i, np.ones([i.shape[0], pad,]+list(i.shape[2:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.hstack(images)
	

def img_grid(images, nb_images_per_row=10, pad=0, pad_value=255):
	ret = []
	while len(images) >= nb_images_per_row:
		ret.append(img_horizontal_concat(images[0:nb_images_per_row], pad_bottom=True, pad=pad, pad_value=pad_value))
		images = images[nb_images_per_row:]
	if len(images) != 0:
		ret.append(img_horizontal_concat(images, pad_bottom=True, pad=pad, pad_value=pad_value))
	return img_vertical_concat(ret, pad=pad, pad_right=True, pad_value=pad_value)






class DirtyLabelPoisonedDataset(BaseOperationDataset):
	"""
		used in training
	"""

	def __init__(self, dataset, trigger : Trigger, num_poison_images : int, target_class : int, source_classes = None,
			 seed=0, sample_save_path=None):
		"""
			dataset : the dataset to be copied,  triggers are injected into the copied dataset.
			trigger : instance of Trigger
			target_class : 
			poison_ratio : the probility of inserting triggers
		"""

		self._trigger = trigger
		self._target_class = target_class
		self._num_poison_images = num_poison_images

		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform
		self.source_classes = source_classes


		print("DirtyLabelPoisonedDataset : ")
		print("data : ", self.data.dtype, self.data.shape, self.data.max(), self.data.min())
		print("targets : ", self.targets.dtype, self.targets.shape, self.targets.max(), self.targets.min())

		np.random.seed(seed)

		if source_classes is None:
			target_img_indices = np.array([i for i, t in enumerate(self.targets) if t != target_class])
		else:
			source_classes = set(source_classes)
			target_img_indices = np.array([i for i, t in enumerate(self.targets) if t in source_classes])

		target_img_indices = np.random.choice(target_img_indices, size=self._num_poison_images, replace=False)

		for ind in target_img_indices:
			self.data[ind] = self._trigger.paste_to_np_img(self.data[ind])
			self.targets[ind] = self._target_class


		if sample_save_path is not None:
			poisoned_data = self.data[np.random.choice(target_img_indices, size=100, replace=False)]
			poisoned_imgs = img_grid(poisoned_data, 10, 1)[:,:,::-1]
			cv2.imwrite(sample_save_path, poisoned_imgs)

		self._poisoned_image_indices = set(target_img_indices)


	def get_name(self, dataset_name):
		pass







class AllPoisonedDataset(BaseOperationDataset):
	"""
		used in training
	"""

	def __init__(self, dataset, trigger : Trigger, target_class : int):
		"""
			dataset : the dataset to be copied,  triggers are injected into the copied dataset.
			trigger : instance of Trigger
			target_class : 
			poison_ratio : the probility of inserting triggers
		"""

		self._trigger = trigger
		self._target_class = target_class

		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform

		for ind in range(len(self.data)):
			self.data[ind] = self._trigger.paste_to_np_img(self.data[ind])
			self.targets[ind] = self._target_class

	def get_name(self, dataset_name):
		pass




