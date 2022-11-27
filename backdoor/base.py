import os
import sys
import numpy as np
import random
from PIL import Image
import time

import copy

from concurrent.futures import ThreadPoolExecutor

import torch
from backdoor.trigger import Trigger




class BaseOperationDataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform

	def get(self, index):
		return self.data[index], self.targets[index]

	def set(self, index, img, target):
		self.data[index] = img
		self.targets[index] = target

	def get_batch(self, indices):
		return self.data[indices], self.targets[indices]

	def set_batch(self, indices, imgs, targets):
		for i, ind in enumerate(indices):
			self.data[ind] = imgs[i]
			self.targets[ind] = targets[i]

	def copy(self, transform=None):
		instance = BaseOperationDataset(self)
		if transform is not None:
			instance.transform = transform
		return instance

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target




class SimpleDataset(BaseOperationDataset):
	def __init__(self, data, targets, transform):

		print("SimpleDataset : ")
		print("data : ", data.shape, data.dtype, data.max(), data.min())
		print("targets : ", targets.shape, targets.dtype, targets.max(), targets.min())
		self.data = np.array(data, copy=True)
		self.targets = np.array(targets, copy=True)
		self.transform = transform
		

	def __len__(self):
		return len(self.data)



class SimpleSubset(BaseOperationDataset):
	def __init__(self, dataset, indices):
		self.data = np.array(dataset.data, copy=True)[indices]
		self.targets = np.array(dataset.targets, copy=True)[indices]
		self.transform = dataset.transform

	def __len__(self):
		return len(self.data)





class ImageFolderDataset(BaseOperationDataset):

	def __init__(self, folder_path, preprocess_before, transform):

		classes = os.listdir(folder_path)
		classes = sorted(classes)

		image_path_list = []
		image_label_list = []
		for ind, cls_name in enumerate(classes):
			file_list = [os.path.join(folder_path, cls_name, f) for f in os.listdir(os.path.join(folder_path, cls_name))]
			image_path_list += file_list
			image_label_list += [ind,]*len(file_list)

		def get_img(fp):
			img = Image.open(fp)
			img = preprocess_before(img)
			img = np.array(img)
			return img

		start = time.time()
		with ThreadPoolExecutor(max_workers=10) as pool:
			image_list = [img for img in pool.map(get_img, image_path_list)]
			print('--------------')
		end = time.time()
		print("Read dataset time elapse : {:02f}s".format(end-start))
		
		self.data = np.array(image_list, copy=True)
		self.targets = np.array(image_label_list, copy=True)
		self.transform = transform


	def get_stat(self):
		print(self.data.shape)
		print(self.data.dtype)
		data = self.data.astype(np.float) / 255.0
		print("Mean : ", np.mean(data, axis=(0, 1, 2)))
		print("Std  : ", np.std(data, axis=(0, 1, 2)))






class TriggerPastedTestDataset(BaseOperationDataset):

	"""
		used in test the effect of trigger
	"""

	def __init__(self, dataset, trigger : Trigger, target_class:int):
		"""
			dataset : the dataset to be copied,  triggers are injected into the copied dataset.
			trigger : instance of Trigger
			target_class : 
			poison_ratio : the probility of inserting triggers
		"""

		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform
		self._trigger = trigger
		self._target_class = target_class

	@property
	def target_class(self):
		return self._target_class

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img_t = self._trigger.paste_to_np_img(img)
		img = Image.fromarray(img)
		img_t = Image.fromarray(img_t)

		if self.transform is not None:
			img = self.transform(img)
			img_t = self.transform(img_t)

		return img, img_t, target

