import os
import sys


import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import pickle as pkl

import torch
from torchvision.datasets import CIFAR10
import cv2
from tqdm import tqdm


import math
import torch.nn.functional as F


from backdoor.base import TriggerPastedTestDataset


from backdoor.trigger import MaskedTrigger, Trigger, cifar_triggers
from backdoor.trigger import WaterMarkTrigger
from backdoor.trigger import SteganographyTrigger
from backdoor.trigger import SinusoidalTrigger
from backdoor.base import TriggerPastedTestDataset



def logical_and(logi_list):
	result = logi_list[0]
	for logi in logi_list[1:]:
		result = np.logical_and(result, logi)
	return result


def count_true(res):
	return np.sum(res.astype(np.float32), axis=-1)



def test_model_acc(args, model, test_ds):
	test_loader = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
	_correct = 0
	_sum = 0
	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_label) in enumerate(test_loader):

			predict_y = model(test_x.float().cuda()).detach()
			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(np.equal(predict_ys, test_label))

			_sum += num_samples

	return _correct / _sum



def test_model_acc_and_asr(args, model, poisoned_test_ds : TriggerPastedTestDataset):
	test_loader = torch.utils.data.dataloader.DataLoader(poisoned_test_ds, batch_size=args.test_batch_size, shuffle=False)
	target_class = poisoned_test_ds.target_class
	_correct = 0
	_sum = 0

	_success = 0
	_valid = 0.00001

	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_x_t, test_label) in enumerate(test_loader):

			predict_y = model(test_x.float().cuda()).detach()
			predict_y_t = model(test_x_t.float().cuda()).detach()

			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			predict_ys_t = np.argmax(predict_y_t.cpu().numpy(), axis=-1)

			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(
				np.equal(predict_ys, test_label)
			)

			_sum += num_samples

			_success += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label),
				np.equal(predict_ys_t, target_class)
			]))

			_valid += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label)
			]))

	return _correct / _sum, _success / _valid



def infer_trigger_name_from_path(input_path):
	
	ip_splits = input_path.split("_")

	if "steganography" in ip_splits:
		ind = ip_splits.index("steganography")
		return "_".join(ip_splits[ind:ind+4])

	elif "watermark" in ip_splits:
		ind = ip_splits.index('watermark')
		return "_".join(ip_splits[ind:ind+4])

	elif "sinusoidal" in ip_splits:
		ind = ip_splits.index("sinusoidal")
		return "_".join(ip_splits[ind:ind+4])

	elif "tri1" in ip_splits:
		ind = ip_splits.index("tri1")
		return "_".join(ip_splits[ind:ind+5])

	elif "tri2" in ip_splits:
		ind = ip_splits.index("tri2")
		return "_".join(ip_splits[ind:ind+5])

	else:
		raise ValueError("uknown trigger")



def infer_trigger_from_path_internal(ip_splits, train_ds, img_size):

	if "steganography" in ip_splits:
		ind = ip_splits.index("steganography")
		info = ip_splits[ind+1]
		nb_bits = int(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = SteganographyTrigger(info, nb_bits, img_size=img_size)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	elif "watermark" in ip_splits:
		ind = ip_splits.index('watermark')
		data_ind = int(ip_splits[ind+1])
		opacity = float(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = WaterMarkTrigger(np.array(train_ds.data[data_ind]), opacity=opacity)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	elif "sinusoidal" in ip_splits:
		ind = ip_splits.index("sinusoidal")
		sin_delta = int(ip_splits[ind+1])
		sin_freq = int(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = SinusoidalTrigger(sin_delta, sin_freq)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	else:
		find=False
		for tri_name in ["tri1", "tri2", "tri3", "tri4", "trisq33"]:
			if tri_name in ip_splits:
				ind = ip_splits.index(tri_name)
				size = ip_splits[ind+1]
				target_class = int(ip_splits[ind+2][1:])
				offset_to_right = int(ip_splits[ind+3])
				offset_to_bottom = int(ip_splits[ind+4])
				trigger = Trigger(tri_name+"_"+size, cifar_triggers[tri_name+"_"+size], offset_to_right=offset_to_right, offset_to_bottom=offset_to_bottom)
				ip_splits = ip_splits[:ind] + ip_splits[ind+5:]
				find=True
				break

		if not find:
			raise ValueError("uknown trigger")

	return ip_splits, trigger, target_class



def infer_trigger_from_path(input_path, train_ds, img_size):
	ip_splits = input_path.split("_")

	if "trojansq" in ip_splits:
		ind = ip_splits.index("trojansq")
		target_class = int(ip_splits[ind+2][1:])

		while not os.path.exists(os.path.join(input_path, "trojan_trigger.pkl")):
			input_path = os.path.split(input_path)[0]
		
		mask_np, trigger_np, patch_np = pkl.load(open(os.path.join(input_path, "trojan_trigger.pkl"), "rb"))
		trigger = MaskedTrigger(trigger_np, mask_np)

	else:
		_, trigger, target_class = infer_trigger_from_path_internal(ip_splits, train_ds, img_size)

	return trigger, target_class



def infer_model_from_path(input_path):
	for model in ["_resnet18_", "_resnet34_", "_resnet50_", "_cnn13_", "_wrn161_", "_wrn162_", "_wrn401_", "_wrn402_"]:
		if model in input_path:
			return model[1:-1]




def pack_images(images, col=None, channel_last=False):
	if isinstance(images, (list, tuple) ):
		images = np.stack(images, 0)
	if channel_last:
		images = images.transpose(0,3,1,2) # make it channel first
	assert len(images.shape)==4
	assert isinstance(images, np.ndarray)
	
	N,C,H,W = images.shape

	print("pack_images : ", N,C,H,W)
	if col is None:
		col = int(math.ceil(math.sqrt(N)))
	row = int(math.ceil(N / col))
	pack = np.zeros( (C, H*row, W*col), dtype=images.dtype )
	for idx, img in enumerate(images):
		h = (idx//col) * H
		w = (idx% col) * W
		pack[:, h:h+H, w:w+W] = img
	return pack



def test_generators(args, generators, nz, epoch, output_dir, plotter=None, epoch_chooser=None, stacked_output=False, norm_trans_inv=None):

	def default_epoch_chooser(e):
		return epoch < 20 or epoch % 10 == 0
	if epoch_chooser is None:
		epoch_chooser = default_epoch_chooser

	if isinstance(generators, list):
		generators = {"{}".format(ind+1) : pg for ind, pg in enumerate(generators)}
	assert isinstance(generators, dict)


	assert output_dir is not None

	for _, pg in generators.items():
		pg.eval()

	with torch.no_grad():

		z = torch.randn( (args.test_batch_size, nz), dtype=torch.float32 ).cuda()
		gene_dict = { name:pg(z) for name,pg in generators.items() }

		if norm_trans_inv is not None:
			gene_dict = {name:norm_trans_inv(pert) for name,pert in gene_dict.items()}

		gene_pert_dict = {name:pert.detach().cpu().numpy() for name,pert in gene_dict.items()}

		if plotter is not None:
			if stacked_output:
				plotter.scalar("gene_max", epoch, {name:np.max(gene_pert) for name, gene_pert in gene_pert_dict.items()})
				plotter.scalar("gene_min", epoch, {name:np.min(gene_pert) for name, gene_pert in gene_pert_dict.items()})
				plotter.scalar("gene_mean", epoch, {name:np.mean(gene_pert) for name, gene_pert in gene_pert_dict.items()})
			else:
				for name, gene_pert in gene_pert_dict.items():
					plotter.scalar("gene_max_{}".format(name), epoch, np.max(gene_pert))
					plotter.scalar("gene_min_{}".format(name), epoch, np.min(gene_pert))
					plotter.scalar("gene_mean_{}".format(name), epoch, np.mean(gene_pert))

		if epoch_chooser(epoch):
			gene_pert_dict = {name:pack_images(np.clip(gene_pert, 0.0, 1.0)) for name,gene_pert in gene_pert_dict.items()}
			gene_pert_dict = {name:(gene_pert.transpose([1, 2, 0]) * 255.0).astype(np.uint8)[:, :, ::-1] for name,gene_pert in gene_pert_dict.items()}
			for name, gene_pert in gene_pert_dict.items():
				cv2.imwrite(os.path.join(output_dir, "images", "gene_pert_{}_e{}.jpg".format(name, epoch)), gene_pert)




def get_image_prior_losses_l1(inputs_jit):
	# COMPUTE total variation regularization loss
	diff1 = torch.mean(torch.abs(inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]))
	diff2 = torch.mean(torch.abs(inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]))
	diff3 = torch.mean(torch.abs(inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]))
	diff4 = torch.mean(torch.abs(inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]))
	return diff1 + diff2 + diff3 + diff4



def get_image_prior_losses_l2(inputs_jit):
	# COMPUTE total variation regularization loss
	diff1 = torch.norm(inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:])
	diff2 = torch.norm(inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :])
	diff3 = torch.norm(inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:])
	diff4 = torch.norm(inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:])
	return diff1 + diff2 + diff3 + diff4





